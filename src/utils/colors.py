class Colors:
    # source: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    # HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BRIGHT_YELLO = '\033[93m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'  # turn off color
    # FAIL = '\033[91m'
    # UNDERLINE = '\033[4m'


# @staticmethod
def text_with_color(color, text):
    return str(color) + text + str(Colors.RESET)


blue = lambda text: text_with_color(Colors.BLUE, text)
cyan = lambda text: text_with_color(Colors.CYAN, text)
green = lambda text: text_with_color(Colors.GREEN, text)
red = lambda text: text_with_color(Colors.RED, text)
byello = lambda text: text_with_color(Colors.BRIGHT_YELLO, text)
gray = lambda text: text_with_color(Colors.GRAY, text)
bold = lambda text: text_with_color(Colors.BOLD, text)

print_blue = lambda text: print(text_with_color(Colors.BLUE, text))
print_cyan = lambda text: print(text_with_color(Colors.CYAN, text))
print_green = lambda text: print(text_with_color(Colors.GREEN, text))
print_byello = lambda text: print(text_with_color(Colors.BRIGHT_YELLO, text))
print_red = lambda text: print(text_with_color(Colors.RED, text))
print_gray = lambda text: print(text_with_color(Colors.GRAY, text))
print_bold = lambda text: print(text_with_color(Colors.BOLD, text))
