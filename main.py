from llama_cpp import Llama


my_model_path = "./model/Phi-3.5-mini-instruct-Q4_K_L.gguf"
CONTEXT_SIZE = 1024


chat_model = Llama(model_path=my_model_path,
                    n_ctx=CONTEXT_SIZE, n_threads=6)

def generate_response(user_prompt,
                             max_tokens = 1024,
                             context='',
                             temperature = 0.7,
                             top_p = 0.1,
                             echo = False,
                             stop = ["Q"]):
        full_prompt = f"{context}{user_prompt}\nAssistant:"
       
        response_stream = chat_model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
            stream=True)
        
        output=""
        empty_token_count = 0 
        max_empty_tokens = 5

        print("Assistant: ", end="")
        for chunk in response_stream:
            token = chunk["choices"][0]["text"]

            if token == "\n":
                empty_token_count += 1
                if empty_token_count >= max_empty_tokens:
                    break
            else:
                empty_token_count = 0
            
            print(token, end="", flush=True)
            output += token

        print()
        return output

if __name__ == "__main__":
   
   print("Chat opened. Type 'exit' to quit.\n")


   context = ""
   
   while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    context += f"User: {user_input}\n"
    model_response = generate_response(user_input, context=context)

    context += f"Assistant: {model_response}\n"