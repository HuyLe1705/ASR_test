#Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

#Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

def get_chat_response(query, model="gpt-4-1106-preview"):
    try:
        #Create the chat completion request
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": "You are a expert sentiment analysist."},
                {"role": "user", "content": query}
            ]
        )

        #Extract and return the chat response text
        #Modification: Use the 'completion' variable to access the chat response
        chat_response_text = completion.choices[0].message.content
        return chat_response_text
    
    except Exception as e:
        #Handle any exceptions that occur during the API request
        return f"An error occured: {str(e)}"

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segments.text for segment in segments)
    return transcription

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000/1024*chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf=wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframs(b''.join(frames))
    wf.close()

#Initialize a deque with a maximum length of 100 characters
transcription_buffer = deque(maxlen=100)

def create_ui():
    #Create the main window
    window = tk.Tk()
    window.title("Chat Output Display")

    #Define a custom font
    customFont = font.Font(family="Helvetica", size=30)

    #Create a scrolled text area with custom font
    output_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=30, height=20, font=customFont)
    output_area.pack(padx=10, pady=10)

    #Function to update the output area
    def update_output(text):
        output_area.insert(tk.END, text + '\n')
        output_area.see(tk.END)

    return window, update.output

def transcription_thread(update_output):
    #Choose your model settings
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = "" #Initialize an empty string to accumulate transcriptions

    try: 
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription =transcribe_chunk(model, chunk_file)
            os.remove(chunk_file)

            for char in transcription:
                transcription_buffer.append(char)

            transcription_window = ''.join(transcription_buffer)
            print(NEON_GREEN + transcription + RESET_COLOR)

            user_input = f"Conversation: {transcription_window}. What is the sentiment of the conversation above? (ANSWER ONLY WITH 'POSITIVE', 'NEUTRAL' or 'NEGATIVE')"
            chat_output = get_chat_response(user_input)
            window.after(0, update_output, chat_output)

            #Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")
        #Write the accumulated transcription to the log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main2():
    global window
    window, update_output = create_ui()

    thread = threading.Thread(target=transcription_thread, args=(update_output,))
    thread.daemon = True
    thread.start()

    window.mainloop()

if __name__ == "__main__":
    main2()
