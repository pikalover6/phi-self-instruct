import phi

while (1):

    inp = input().replace("\\n", "\n")
    if (inp):
        print(phi.chat(inp))
    else:
        break
