from openai import OpenAI

class OpenAIBot:
    def __init__(self, engine, api_key="Enter Your API Key Here"):
        # Initialize the OpenAI client with the provided API key
        self.client = OpenAI(api_key=api_key)
        # Initialize conversation with a system message
        self.conversation = [{"role": "system", "content": "You are a helpful assistant."}]
        self.engine = engine
        
    def add_message(self, role, content):
        # Adds a message to the conversation.
        self.conversation.append({"role": role, "content": content})
        
    def generate_response(self, prompt):
        # Add user prompt to conversation
        self.add_message("user", prompt)
        try:
            # Make a request to the API using the chat-based endpoint with conversation context
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=self.conversation,
                store=True
            )
            # Extract the response
            assistant_response = response.choices[0].message.content.strip()
            # Add assistant response to conversation
            self.add_message("assistant", assistant_response)
            # Return the response
            return assistant_response
        except Exception as e:
            error_message = f'Error Generating Response: {str(e)}'
            print(error_message)
            return f"Error: {error_message}"
    
    def clean_json_output(self, output_str):
        """
        Remove triple-backtick fences from model's output to parse valid JSON.
        """
        output_str = output_str.strip()
        if output_str.startswith("```"):
            lines = output_str.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            output_str = "\n".join(lines).strip()
        return output_str
    
    def update_system_prompt(self, system_prompt):
        """
        Updates the system prompt in the conversation history.
        """
        if self.conversation and self.conversation[0]["role"] == "system":
            self.conversation[0]["content"] = system_prompt
        else:
            # Insert a system message at the beginning if not present
            self.conversation.insert(0, {"role": "system", "content": system_prompt})
    
    def reset_conversation(self, keep_system_prompt=True):
        """
        Resets the conversation history, optionally keeping the system prompt.
        """
        system_prompt = "You are a helpful assistant."
        if keep_system_prompt and self.conversation and self.conversation[0]["role"] == "system":
            system_prompt = self.conversation[0]["content"]
        
        self.conversation = [{"role": "system", "content": system_prompt}]