#!/usr/bin/env python3

import time
import tempfile
import wave
from pathlib import Path

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_interfaces.srv import HumanDetected, LLMQuery

from piper.voice import PiperVoice

try:
    from playsound import playsound
    PLAYSOUND_IMPORT_ERROR = None
except Exception as exc:
    playsound = None
    PLAYSOUND_IMPORT_ERROR = exc

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.callback_group = ReentrantCallbackGroup()
        self.srv = self.create_service(HumanDetected,'human_detected', self.handle_human_detected, callback_group=self.callback_group,)
        self.llm_client = self.create_client(LLMQuery,'llm_inference', callback_group=self.callback_group,)
        self.llm_service_wait_timeout = 5.0
        self.llm_response_timeout = 12.0
        self.default_greeting = 'Hello! I hope you are doing well today.'
        self.greeting_prompt = (
            'Reply with exactly one short friendly greeting to a factory worker that you met. You know the worker. Be creative.'
            'No quotes. Maximum 12 words.'
        )
        self.cached_greeting = None

        self.declare_parameter('piper_model_path', '~/piper_models/en_US-lessac-medium/en_US-lessac-medium.onnx')
        model_path = self.get_parameter('piper_model_path').get_parameter_value().string_value
        model_path = str(Path(model_path).expanduser())
        self.voice = None

        if model_path and Path(model_path).is_file():
            self.voice = PiperVoice.load(model_path)
            self.get_logger().info(f'Loaded Piper model: {model_path}')
        else:
            self.get_logger().warn('Piper model path is invalid. Set parameter "piper_model_path" to a valid .onnx model file to enable TTS.')

    def speak(self, text):
        if self.voice is None:
            self.get_logger().warn('TTS skipped: Piper model is not loaded.')
            return False

        if playsound is None:
            self.get_logger().warn(f'TTS skipped: playsound unavailable ({PLAYSOUND_IMPORT_ERROR})')
            return False

        tmp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as tmp_file:
                tmp_wav_path = tmp_file.name

            if hasattr(self.voice, 'synthesize_stream_raw'):
                sample_rate = getattr(getattr(self.voice, 'config', None), 'sample_rate', 22050)
                with wave.open(tmp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    for audio_bytes in self.voice.synthesize_stream_raw(text):
                        wav_file.writeframes(audio_bytes)

                playsound(tmp_wav_path)
                return True

            if hasattr(self.voice, 'synthesize'):
                audio_chunks = list(self.voice.synthesize(text))
                if not audio_chunks:
                    self.get_logger().warn('TTS skipped: Piper returned no audio chunks.')
                    return False

                first_chunk = audio_chunks[0]
                with wave.open(tmp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(first_chunk.sample_channels)
                    wav_file.setsampwidth(first_chunk.sample_width)
                    wav_file.setframerate(first_chunk.sample_rate)
                    for audio_chunk in audio_chunks:
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)

                playsound(tmp_wav_path)
                return True

            self.get_logger().error('TTS skipped: unsupported PiperVoice API (no synthesize methods found).')
            return False
        except Exception as exc:
            self.get_logger().error(f'TTS playback failed: {exc!r}')
            return False
        finally:
            if tmp_wav_path:
                try:
                    Path(tmp_wav_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def capture_voice(self):
        # Simulate voice capture and conversion to text
        simulated_text = "Hello, how are you?"
        self.get_logger().info(f'Captured voice: {simulated_text}')
        return simulated_text

    def sanitize_response_text(self, text):
        cleaned = text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return cleaned or self.default_greeting

    def response_callback(self, msg):
        self.get_logger().info(f'LLM Response: {msg.data}')

    def get_llm_response(self, prompt):
        if not self.llm_client.wait_for_service(timeout_sec=self.llm_service_wait_timeout):
            self.get_logger().warn(f'LLM service unavailable after {self.llm_service_wait_timeout:.1f}s, using fallback response.')
            return self.default_greeting

        llm_request = LLMQuery.Request()
        llm_request.prompt = prompt
        future = self.llm_client.call_async(llm_request)

        deadline = time.monotonic() + self.llm_response_timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:  # wait for response from LLM with timeout
            time.sleep(0.05)

        if not future.done():
            future.cancel()
            self.get_logger().warn(f'LLM request timed out after {self.llm_response_timeout:.1f}s, using fallback response.')
            return self.default_greeting

        try:
            llm_result = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to call LLM service: {exc}')
            return self.default_greeting

        if llm_result is None or not llm_result.response.strip():
            self.get_logger().warn('LLM returned an empty response, using fallback response.')
            return self.default_greeting

        return self.sanitize_response_text(llm_result.response)

    def handle_human_detected(self, request, response):
        if request.prefetching:
            self.get_logger().info('Human detected: prefetching LLM greeting only.')
            if self.cached_greeting is None:
                self.get_logger().info('Generating greeting with LLM...')
                varied_prompt = f'{self.greeting_prompt} Variation token: {time.time_ns() % 1000000}.'
                llm_response = self.get_llm_response(varied_prompt)
                self.cached_greeting = self.sanitize_response_text(llm_response)
            else:
                self.get_logger().info('Using cached greeting.')

            response.response_text = self.cached_greeting
            self.get_logger().info(f'Prefetched greeting: {response.response_text}')
            return response

        self.get_logger().info('Goal reached: playing prefetched greeting.')
        if self.cached_greeting is None:
            self.get_logger().warn('No cached greeting available, generating now.')
            varied_prompt = f'{self.greeting_prompt} Variation token: {time.time_ns() % 1000000}.'
            llm_response = self.get_llm_response(varied_prompt)
            self.cached_greeting = self.sanitize_response_text(llm_response)

        self.get_logger().info(f'Greeting to play: {self.cached_greeting}')
        playback_succeeded = self.speak(self.cached_greeting)
        response.response_text = self.cached_greeting if playback_succeeded else ''
        self.cached_greeting = None

        return response


def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()