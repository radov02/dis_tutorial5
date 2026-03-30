#!/usr/bin/env python3

import json
import time
from urllib import error, request as urllib_request

import rclpy
from rclpy.node import Node
from robot_interfaces.srv import LLMQuery


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.create_service(LLMQuery, 'llm_inference', self._handle_query)
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('model_name', 'llama3.2:3b')
        self.declare_parameter('request_timeout', 15.0)
        self.declare_parameter('max_tokens', 24)
        self.declare_parameter('temperature', 0.2)
        self.declare_parameter('keep_alive', '15m')

        self.ollama_url = self.get_parameter('ollama_url').value
        self.model_name = self.get_parameter('model_name').value
        self.request_timeout = float(self.get_parameter('request_timeout').value)
        self.max_tokens = int(self.get_parameter('max_tokens').value)
        self.temperature = float(self.get_parameter('temperature').value)
        self.keep_alive = self.get_parameter('keep_alive').value
        self.get_logger().info(f'LLMNode initialized with model "{self.model_name}", timeout={self.request_timeout:.1f}s, max_tokens={self.max_tokens}, keep_alive={self.keep_alive}')

    def _handle_query(self, request, response):
        prompt = request.prompt.strip()
        self.get_logger().info(f'Received LLM query: {prompt}')

        if not prompt:
            response.response = "Prompt is empty."
            self.get_logger().warn("Received an empty prompt")
            return response
        
        # Send prompt to local Ollama
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }).encode('utf-8')
        http_request = urllib_request.Request(
            self.ollama_url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            request_start = time.monotonic()
            with urllib_request.urlopen(http_request, timeout=self.request_timeout) as http_response:
                response_data = json.loads(http_response.read().decode('utf-8'))
            elapsed = time.monotonic() - request_start
            generated_text = response_data.get('response', '').strip()
        except TimeoutError:
            response.response = "LLM request timed out."
            self.get_logger().error("Timed out while waiting for Ollama")
            return response
        except error.HTTPError as exc:
            response.response = "LLM failed to generate a response."
            self.get_logger().error(f"Ollama returned HTTP {exc.code}")
            return response
        except error.URLError as exc:
            response.response = "LLM is unavailable. Check that Ollama is running."
            self.get_logger().error(f"Failed to call Ollama: {exc}")
            return response
        except json.JSONDecodeError:
            response.response = "LLM returned an invalid response."
            self.get_logger().error("Failed to parse Ollama JSON response")
            return response

        if not generated_text:
            response.response = "LLM returned an empty response."
            self.get_logger().warn("Ollama returned an empty response")
            return response

        response.response = generated_text
        self.get_logger().info(f'Generated LLM response in {elapsed:.2f}s: {generated_text}')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()