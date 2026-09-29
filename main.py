from services.college_notes_assistant import CollegeNotesAssistant
def main():
    print("="*60)
    print("College Notes AI Assistant")
    print("="*60)

    pdf_path = input("\nEnter the path to the PDF file: ")
    college_notes_instance = CollegeNotesAssistant(pdf_path)

    print("\nYou can ask questions about your notes.")
    print("Type 'explain <concept>' to explain a concept.")
    print("Type 'summarize <topic>' to summarize a topic.")
    print("Type 'define <term>' to find a definition.")
    print("Type 'debug on/off' to toggle debug output.")
    print("Type 'q' to quit.\n")

    while True:
        user_input = input("Enter your query (or type 'q' to quit): ").strip()
        if user_input == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        if user_input.lower().startswith("debug "):
            mode = user_input.split()[1].lower()
            if mode == "on":
                college_notes_instance.show_debug = True
                print("Debug mode is ON.\n")
            elif mode == "off":
                college_notes_instance.show_debug = False
                print("Debug mode is OFF.\n")
                continue
            else:
                print("Invalid debug command. Using default condition (debug off).\n")

        elif user_input.lower().startswith("summarize "):
            topic = user_input[10:].strip()
            try:
                if topic == "":
                    raise ValueError("Topic cannot be empty. Please provide a valid topic to summarize.")
                response = college_notes_instance.summarize_chat(topic)
                if response is None:
                    raise ValueError("No response generated for the topic. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue

        elif user_input.lower().startswith("define "):
            term = user_input[7:].strip()
            try:
                if term == "":
                    raise ValueError("Term cannot be empty. Please provide a valid term to define.")
                response = college_notes_instance.define_chat(term)
                if response is None:
                    raise ValueError("No response generated for the term. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue

        elif user_input.lower().startswith("explain "):
            concept = user_input[8:].strip()
            try:
                if concept == "":
                    raise ValueError("Concept cannot be empty. Please provide a valid concept to explain.")
                response = college_notes_instance.explain_chat(concept)
                
                if response is None:
                    raise ValueError("No response generated for the concept. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue
            
            
        else:
            print("Processing your query...")
            response = college_notes_instance.chat_with_pdf(user_input)
            print(f"Response: {response}\n")
            continue
if __name__ == "__main__":
    main()