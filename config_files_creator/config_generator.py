import itertools
import json

norm = "LHSCTMRP"


def get_all_nargs():
	total = []
	for i in range(len(norm) + 1):
		total += list(itertools.combinations(norm, i + 1))
	return total + [('', '')]


def handle_features_input():
	features = []
	choose = input("Enter selected Feature:\n\t1.ngrams\n")
	if choose == '1':  # ngrams
		print("Enter in this template: *type* *n* *k* *num features*\nexample: w 1 0 1000\n")
		tem = None
		while True:
			tem = input("enter more features (optional)\n")
			if not (tem != '' and tem != '\n'):
				break
			parts = tem.split(" ")
			type = parts[0]
			count = parts[3]
			n = parts[1]
			k = parts[2]
			tfidf = input("tfidf?(y/n)\n")
			if tfidf.lower() == 'y':
				tfidf = "tfidf"
			else:
				tfidf = 'tf'
			features.append("ngrams_{}_{}_{}_{}_{}".format(count, type, tfidf, n, k))
	return features


def handle_methods_input():
	methods = []
	headers = input("Enter first letter of selected methods:\nmlp(m),svc(s),rf(r),lr(l),mnb(n)\nex: msr,srl\n")
	headers = headers.lower()
	if 'm' in headers:
		methods.append('mlp')
	if 's' in headers:
		methods.append('svc')
	if 'r' in headers:
		methods.append('rf')
	if 'l' in headers:
		methods.append('lr')
	if 'n' in headers:
		methods.append('mnb')
	return methods


if __name__ == '__main__':
	train = input("Enter training path:\n")
	test = input("Enter testing path:(optional)\n")
	output_path = input("Enter output path:\n")
	features = handle_features_input()
	methods = handle_methods_input()
	results = input("Enter results path:\n")
	write_dir = input("Enter where to dump all config files:\n")
	nargs = get_all_nargs()
	for n in nargs:
		data = {"train": train, "test": test, "output_csv": output_path, "nargs": ''.join(sorted(list(n))), "features": features,
				"results": results, "methods": methods}
		if n != ('', ''):
			name = write_dir + "\\" + ''.join(sorted(list(n))) + ".json"
		else:
			name = write_dir + "\\" + "NONE.json"
		with open(name, "w+") as file:
			json.dump(data, file,indent=4)
