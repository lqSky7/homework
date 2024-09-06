# A farmer with a fox, a goose, and a sack of corn needs to cross a river. Now he is on the east side of the river and wants to go to west side. The farmer has a rowboat, but there is room for only the farmer and one of his three items. Unfortunately, both the fox and the goose are hungry. The fox cannot be left alone with the goose, or the fox will eat the goose. Likewise, the goose cannot be left alone with the sack of corn, or the goose will eat the corn. Given a sequence of moves find if all the three items fox, goose and corn are safe. The input sequence indicate the item carried by the farmer along with him in the boat. ‘F’ – Fox, ‘C’ – Corn, ‘G’ – Goose, N-Nothing. As he is now on the eastern side the first move is to west and direction alternates for each step.​


east = ['farmer', 'fox', 'goose', 'corn']
west = []
boat = []

for i in east:
    
