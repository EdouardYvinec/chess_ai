import cv2
import numpy


class KillApp(Exception):
    pass


class OpponentSelector:
    img_dim = 1080
    bbox = [510, 70]

    options = [
        "random",
        "stockfish (400)",
        "stockfish (850)",
        "stockfish (1000)",
        "stockfish (1400)",
        "stockfish (2000)",
    ]

    def __init__(self) -> None:
        self.mouseX = 0
        self.mouseY = 0
        self.clicked = False

    def add_box(self, image: numpy.array, text: str, y: int) -> numpy.array:
        x = self.img_dim // 2 - self.bbox[0] // 2
        image = cv2.rectangle(image, (x, y), (x + self.bbox[0], y + self.bbox[1]), (36, 255, 12), 1)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0][0]
        cv2.putText(
            image,
            text,
            (x + (self.bbox[0] - size) // 2, y + self.bbox[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (36, 255, 12),
            2,
        )
        return image

    def create_menu(self) -> numpy.array:
        img = numpy.zeros((self.img_dim, self.img_dim, 3), dtype=numpy.uint8)
        for cpt, option in enumerate(self.options):
            img = self.add_box(image=img, text=option, y=400 + (self.bbox[1] + 30) * cpt)
        return img

    def check_valid_click(self) -> bool:
        x_valid = (self.mouseX >= self.img_dim // 2 - self.bbox[0] // 2) and (
            self.mouseX <= self.img_dim // 2 + self.bbox[0] // 2
        )
        y_valid = False
        for cpt in range(len(self.options)):
            y = 400 + 100 * cpt
            y_valid = y_valid or ((self.mouseY >= y) and (self.mouseY <= y + self.bbox[1]))
        return x_valid and y_valid

    def window(self) -> str:
        def mouse_coordinates(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked = True
                self.mouseX, self.mouseY = x, y

        cv2.namedWindow("chess mini app")
        cv2.setMouseCallback("chess mini app", mouse_coordinates)
        while not self.clicked:
            cv2.imshow("chess mini app", self.create_menu())
            key = cv2.waitKey(1)
            if key in [27, 113]:
                raise KillApp
            if self.clicked:
                self.clicked = False
                if self.check_valid_click():
                    break
        return self.options[self.mouseY // 100 - 4]

    def select(self) -> str:
        try:
            opponent = self.window()
        except KillApp:
            return ""
        return opponent
