import random

print('Welcome to hand cricket in computer version')

playre_name = str(input('Enter your name: '))


def letsplay(n):
    out = False
    runs = 0
    while out==False and n>0:
        hit = int(input("Enter ur choice of score between 1 to 6: "))
        random_score = random.randint(1,6)
        if hit==random_score:
            out = True
            print("Sorry you got out")
        else:
            runs+=hit
        n-=1
        print("Current score is {}".format(runs))
            
    return runs

n = int(input('How many balls you want to play'))
runs = letsplay(n)

print("You scored: ",end='')
print(runs)