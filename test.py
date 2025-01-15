import pandas as pd

class FileSaver:
    def __init__(self, fail_to_recourse, overall_acc_list, jsd_list, avgRecourseCost, avgNewRecourseCost, avgOriginalRecourseCost):
        self.failToRecourse = fail_to_recourse
        self.overall_acc_list = overall_acc_list
        self.jsd_list = jsd_list
        self.avgRecourseCost = avgRecourseCost
        self.avgNewRecourseCost = avgNewRecourseCost
        self.avgOriginalRecourseCost = avgOriginalRecourseCost

    def save_to_csv(self, recourse_num, threshold, acceptance_rate, cost_weight, dataset):
        filename = f"{recourse_num}_{threshold}_{acceptance_rate}_{cost_weight}_{dataset}.csv"
        
        data = {
            'failToRecourse': self.failToRecourse,
            'acc': self.overall_acc_list,
            'jsd': self.jsd_list,
            'avgRecourseCost': self.avgRecourseCost,
            'avgNewRecourseCost': self.avgNewRecourseCost,
            'avgOriginalRecourseCost': self.avgOriginalRecourseCost
        }
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"File saved as: {filename}")

