import sys

stdin_input = sys.stdin.readline()

while stdin_input != "end":
    print("Python Child Received:", stdin_input)
    sys.stdout.flush()
    stdin_input = sys.stdin.readline()

print("Python Child Ended")
sys.stdout.flush()
