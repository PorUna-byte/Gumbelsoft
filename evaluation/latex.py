import os
import json
import re
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--diverse_path', type=str, default="com_diverse.txt")
    parser.add_argument('--detect_path', type=str, default="table1_ana.txt")
    parser.add_argument('--task', type=str, default="Com")
    return parser

args = get_args_parser().parse_args()

with open(os.path.join(args.experiments_path, args.diverse_path), 'r') as file:
    diverse_content = file.read()

# Next, I will read the contents of 'table1_ana.txt' to understand its structure and data.
with open(os.path.join(args.experiments_path, args.detect_path), 'r') as file:
    detect_content = file.read()
    

detect_data = json.loads(detect_content)
# The structure seems to be in a pattern with method names followed by their respective metrics.

# Regex pattern to extract the method name and its metrics
pattern = r"([\w\d\.]+):\nAverage bleu is:(\d+\.\d+)\nAverage Dist-1 is:(\d+\.\d+)\nAverage Dist-2 is:(\d+\.\d+)"

# Extracting data using regex
diverse_data = re.findall(pattern, diverse_content)

# Converting the extracted data into a dictionary for easier access and manipulation
diverse_dict = {method: {"Self-Bleu": float(bleu), "Dist-1": float(dist1), "Dist-2": float(dist2)} 
                    for method, bleu, dist1, dist2 in diverse_data}

# Function to format the LaTeX row for each method
def format_latex_row(method, diverse_data, detect_data):
    # Extracting data from both dictionaries
    self_bleu = diverse_data.get('Self-Bleu', 'N/A')
    dist_1 = diverse_data.get('Dist-1', 'N/A')
    dist_2 = diverse_data.get('Dist-2', 'N/A')
    auroc = detect_data.get('Avg_AUROC', {}).get('tok_100', 'N/A')
    fpr = detect_data.get('Avg_FPR', {}).get('tok_100', 'N/A')
    fnr = detect_data.get('Avg_FNR', {}).get('tok_100', 'N/A')
    ppl = detect_data.get('Avg_ppl', 'N/A')

    # Formatting the LaTeX row
    return f"{method} & {self_bleu} & {dist_1} & {dist_2} & {auroc} & {fpr} & {fnr} & {ppl} \\\\\n"

# Creating the updated LaTeX table
latex_table = "\\begin{table*}[htbp]\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.2}\n\\centering\n\\begin{tabular}{|l|ccc|ccc|c|}\n\\hline\nMethod & \\multicolumn{3}{c|}{Diversity} & \\multicolumn{3}{c|}{Detectability} & Quality \\\\\n & Self-Bleu $\\downarrow$ & Dist-1 $\\uparrow$ & Dist-2 $\\uparrow$ & AUROC $\\uparrow$ & FPR $\\downarrow$ & FNR $\\downarrow$ & ppl $\\downarrow$ \\\\\n\\hline\n"

# Adding rows to the LaTeX table
for method, diverse_metrics in diverse_dict.items():
    detect_metrics = detect_data.get(method, {})
    latex_table += format_latex_row(method, diverse_metrics, detect_metrics)

# Closing the table
latex_table += "\\hline\n\\end{tabular}\n\\caption{We performed a comprehensive comparative analysis to evaluate the diversity of different variations of the exponential watermark technique. The detectability of these variations was assessed using the Llama2-7b-chat and the Alpaca datasets. Meanwhile, we evaluated their diversity on the Llama2-7b-chat and our specially curated dataset. This experiment was replicated five times, and we present the average results obtained from these iterations.}\n\\label{table:diversity_com}\n\\end{table*}"

# latex_table = latex_table.replace("openaiNg","")
# latex_table = latex_table.replace("gumbelsoftNg","")
# latex_table = latex_table.replace("_","")
with open(os.path.join(args.experiments_path, f"latex_code_{args.task}.txt"),'w') as f:
    f.write(latex_table)



