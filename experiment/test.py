import re
conversation = "[conversation] [user] can you find me a music event on 9th of this month? i prefer pop music [system] what city should i search in [user] please search in philadelphia [system] i found 3 events you may like. what about conan gray at the fillmore philadelphia on march 9th at 6:30 pm [user] ok, what else can you find for me [system] what about girl in red at the foundry on march 9th at 6 pm [user] what is the venue address [system] they are located in 29 east allen street 2nd floor [user] perfect. now please check my savings account balance"
conversation = "[user] Actually, can you make it three tickets? [system] okei, all booked"
history = "[history] u228, u228, u228, u228; s151; u228; s78, s78, s78, s78, s78; u61; s78, s78, s78, s78; u150; s78; u54, u66, u232"
user_turns = re.findall(r'(?:\[user\](.*?)\[system\])', conversation)
system_turns = re.findall(r'(?:\[system\](.*?)(?:\[user\]|$))',  conversation)
history_turns = re.findall(r'((?:\w+, )*\w+)(?:;|\Z)', history)
print(user_turns)
print(system_turns)
print(history_turns, len(history_turns))
conversation_turns = re.findall(r'((?:\[user\]|\[system\]).*?)(?=$| \[user\]| \[system\])',  conversation)

print("conversations", conversation_turns, len(conversation_turns))