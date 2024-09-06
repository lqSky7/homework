east = ["farmer", "fox", "goose", "corn"]
west = []

def move(item, farmer):
    if item in east and farmer in east:
        east.remove(item)
        east.remove(farmer)
        west.append(item)
        west.append(farmer)
    elif item in west and farmer in west:
        west.remove(item)
        west.remove(farmer)
        east.append(item)
        east.append(farmer)

def move_farmer_back(farmer):
    if farmer in east:
        east.remove(farmer)
        west.append(farmer)
    elif farmer in west:
        west.remove(farmer)
        east.append(farmer)

def safe_check():
    if "farmer" not in east and ("goose" in east and "corn" in east or "goose" in east and "fox" in east):
        return False
    if "farmer" not in west and ("goose" in west and "corn" in west or "goose" in west and "fox" in west):
        return False
    return True

def solve_puzzle():
    moves = 0
    while west != ["farmer", "fox", "goose", "corn"]:
        moves += 1
        if moves > 100:
            return
        
        for i in east + west:
            if i == "farmer":
                continue
            if (i in east and "farmer" in east) or (i in west and "farmer" in west):
                move(i, "farmer")
                if safe_check():
                    print(f"Move {moves} - East:", east, "West:", west)
                    move_farmer_back("farmer")
                    print(f"Move {moves+1} - East:", east, "West:", west)
                    break
                else:
                    move(i, "farmer")  # Move back if not safe
        else:
            move_farmer_back("farmer")
            print(f"Move {moves} - East:", east, "West:", west)

solve_puzzle()
print("Final state - East:", east, "West:", west)