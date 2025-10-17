

#USER INTERFACE
def chatbot():

    print("👋 Welcome to CarePredictAI!")
    name = input("Please enter your name: ")
    print(f"I'm sorry you're feeling unwell, {name}. Let's check what might be going on.")

    age = int(input("Enter your age: "))
    gender = input("Enter your gender (Male/Female): ")
    history = input("Briefly describe your medical history: ")
    lifestyle = input("Describe your lifestyle (e.g., active, sedentary): ")
    symptoms_text = input("Describe your symptoms: ")