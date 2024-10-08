import tkinter as tk
import pandas as pd

from tkinter import filedialog

from  .split import split_into_sentences


class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Labeling Tool")

        self.sentences = []
        self.index = 0
        self.data = []

        # UI Elements
        self.text_label = tk.Label(root, text="", wraplength=500)
        self.text_label.pack(pady=20)

        self.button_0 = tk.Button(root, text="Label 0 (Novel Content)", command=self.label_0, width=20)
        self.button_0.pack(side=tk.LEFT, padx=20)

        self.button_1 = tk.Button(root, text="Label 1 (Non-Novel Content)", command=self.label_1, width=20)
        self.button_1.pack(side=tk.RIGHT, padx=20)

        self.load_button = tk.Button(root, text="Load Text File", command=self.load_file, width=20)
        self.load_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Dataset", command=self.save_dataset, width=20)
        self.save_button.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                self.sentences = split_into_sentences(text)
                self.index = 0
                self.update_text()

    def update_text(self):
        if self.index < len(self.sentences):
            self.text_label.config(text=self.sentences[self.index])
        else:
            self.text_label.config(text="All sentences labeled!")

    def label_0(self):
        if self.index < len(self.sentences):
            self.data.append({"text": self.sentences[self.index], "label": 0})
            self.index += 1
            self.update_text()

    def label_1(self):
        if self.index < len(self.sentences):
            self.data.append({"text": self.sentences[self.index], "label": 1})
            self.index += 1
            self.update_text()

    def save_dataset(self):
        if self.data:
            df = pd.DataFrame(self.data)
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                df.to_csv(save_path, index=False)
                self.text_label.config(text="Dataset saved!")
        else:
            self.text_label.config(text="No data to save!")


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
