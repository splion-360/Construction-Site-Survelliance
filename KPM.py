import pygame
def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))

def getkey(key):
    ans = False
    for eve in pygame.event.get():
        pass
    keyinput = pygame.key.get_pressed()
    mykey = getattr(pygame,'K_{}'.format(key))
    if keyinput[mykey]:
        ans = True
    pygame.display.update()
    return ans

def main():
    if getkey("LEFT"):
        print("L")
    if getkey("RIGHT"):
        print("R")

if __name__ == "__main__":
    init()
    while True:
        main()