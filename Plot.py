import matplotlib.pyplot as plt
from Parsers import CSVParser

class Plot:
    def __init__(self, csv: str) -> None:
        self.parser = CSVParser(csv)
        self.test_instance = self.parser.getColumnFloat("test_instance")
        self.label = self.parser.getColumnFloat("label")
    
    def plotK(self, k: int) -> None:
        difference = []
        for a, b in zip(self.label, self.parser.getColumnFloat(str(k))):
            difference.append(a - b)
        
        plt.bar(self.test_instance, difference)
        plt.savefig(f"figs/plot_{k}.jpg")

if __name__ == "__main__":
    test = Plot("results.csv")
    test.plotK(1)
    test.plotK(3)
    test.plotK(5)
    test.plotK(10)
    test.plotK(50)
    test.plotK(100)
    test.plotK(500)
    test.plotK(1000)

