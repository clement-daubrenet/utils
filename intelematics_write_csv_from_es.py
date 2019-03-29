import os
import json
import csv


for file in os.listdir('.'):
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)
            csv_file = file.split('.')[0] + '.csv'
            with open(csv_file, 'w') as csvfile:
                for header in sorted(data['hits']['hits'][0]['_source'].keys()):
                    csvfile.write(header + ',')
                csvfile.write('\n')
                for line in data['hits']['hits']:
                    for key in sorted(line['_source'].keys()):
                        print(key)
                        csvfile.write(str(line['_source'][key]) + ',')
                    csvfile.write('\n')


            # with open(csv_file, 'wb') as csvfile:
            #     spamwriter = csv.writer(csvfile, delimiter=',')
            #     for line in data:
            #         print(line)
                # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
            # filewriter = csv.writer(
            #     file.split('.')[0] + '.csv', quotechar=',')
            # for field in data:
            #     print(field)
                # col1 = hit["some"]["deeply"]["nested"]["field"].decode(
                #     'utf-8')  # replace these nested key names with your own
                # col1 = col1.replace('\n', ' ')