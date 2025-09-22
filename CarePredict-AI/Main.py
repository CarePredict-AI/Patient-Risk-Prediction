def intro():
    print("HELLO! \nWELCOME TO CarePredict-AI\n")

    while True:
       response = input("DO YOU WANT TO GO TO ADMIN OR PATIENT SECTION\n").upper()

       if response == "ADMIN":
           print("SECTION IS OFFLINE")
           break

       elif response == "PATIENT":
           print("SECTION IS OFFLINE")
           break

       else:
           print("CHOOSE BETWEEN ADMIN OR PATIENT\n")

intro()