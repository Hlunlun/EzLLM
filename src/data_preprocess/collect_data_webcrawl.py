import re
import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry


ENZYME_TYPES = ['protease','amylase','lipase','mannanase','cellulase', 'pectinase', 'others']

def crawl_brenda_ecnumber_table(url,save_path='ecnumber_brenda.csv'):

    # set headers 
    brenda_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

    # get web
    response = requests.get(url, headers=brenda_headers)
    response.raise_for_status()  # check if successful

    # parse html
    soup = BeautifulSoup(response.content, 'html.parser')

    # get <table>
    table = soup.find('table')

    # get <table> rows
    rows = table.find_all('tr')

    # get header of table
    headers = [header.get_text(strip=True) for header in rows[0].find_all('th')]  + ['Long details', 'Short details']
    headers.remove('Show details')

    base_url = 'https://www.brenda-enzymes.org'
    # get data in table
    data = []
    for row in rows[1:]:
        cells = row.find_all('td')
        row_data = []
        for cell in cells:
            # find links in <a>
            links = cell.find_all('a')   
            links_new = []     
            if links:
                long_url, short_url = [
                    base_url + (href.get('href').replace('.', '', 1) if href.get('href')[0] == '.' else href.get('href'))
                    for href in links
                ]
                row_data.extend([long_url, short_url])
            else:
                row_data.append(cell.get_text(strip=True))
        data.append(row_data)
        
    # transfer to pandas series
    df = pd.DataFrame(data, columns=headers)

    print(df.head())

    # save as csv file
    df.to_csv(save_path, index=False)
    print(f'save ecnumber from brenda to {save_path}')


def crawl_brenda_table_by_ec(ec_num, organism, table='temperature'):
    pH_options = ['pH_Optimum', 'pH_Range', 'pH_Stability']
    pH_num = [45,46,47]
    temperature_options = ['Temperature_Optimum', 'Temperature_Range', 'Temperature_Stability']
    temperature_num = [41,42,43]

    tables = pH_options if table == 'pH' else temperature_options
    table_num = pH_num if table == 'pH' else temperature_num

    data = {}

    for table in tables:
        url = f'https://www.brenda-enzymes.org/all_enzymes.php?ecno={ec_num}&table={table}#TAB'
       
        # set headers 
        brenda_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        # get web
        response = requests.get(url, headers=brenda_headers)
        response.raise_for_status()  # check if successful
        # parse html
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.find(id=f'tab{table_num[tables.index(table)]}')

        if rows == None:   # 2024/08/03 LUN check if information exists
            continue
        
        for row in rows.find_all('div'):
            cells = row.find_all('div', class_='cell')

            if len(cells) > 1 and cells[1].text.strip() in organism:
                value = cells[0].text.strip()  # 提取第一個單元格中的值
                data[table] = value
                break
    return data



re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)



def crawl_uniprot_by_ecnumber(ec_num, enzyme_type='protease', download=True,
    save_path = '', file_name='',
    fields = ['accession','id', 'protein_name', 'gene_names', 'organism_name', 'length', 'cc_function', 'cc_biotechnology','cc_interaction'],
    headers = ['Entry', 'Entry Name', 'Protein names', 'Gene Names', 'Organism', 'Length', 'Function [CC]',	'Biotechnological use', 'Interacts with'],
    return_format = 'tsv',
    reviewed = 'true'):        

    # url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Ccc_interaction&format=tsv&query=(ec:1.1.1.44)%20AND%20%28reviewed%3Atrue%29&size=500'
    # url = 'https://rest.uniprot.org/uniprotkb/search?fields=comment_count,feature_count,length,structure_3d,annotation_score,protein_existence,lit_pubmed_id,accession,organism_name,protein_name,gene_names,reviewed,keyword,id%2Ccc_interaction&format=tsv&query=(ec:1.1.1.44)%20AND%20(reviewed%3Atrue)&size=500'
    url = f'https://rest.uniprot.org/uniprotkb/search?fields={",".join(fields)}&format={return_format}&query=(ec:{ec_num})%20AND%20(reviewed%3A{reviewed})&size=500'
    
    data = []
        
    
    if return_format == 'tsv':
        if get_batch(url) != None:
            for batch, total in get_batch(url):
                for line in tqdm(batch.text.splitlines()[1:]):
                    data.append(line.split('\t')) 
            df = pd.DataFrame(data, columns=headers)
    
        ec_num_str = '_'.join(ec_num.split('.'))
        reviewed_num = 0 if reviewed=='false' else 1
        file_name = f'{enzyme_type}_ec{ec_num_str}_reviewed{reviewed_num}_{len(df)}.csv' if file_name=='' else file_name
        save_path += file_name
        if download and len(data)>0:
            df.to_csv(save_path, index=False)
            print(f'Save file for ec:{ec_num} with reviewed={reviewed} to path: {save_path}')
    else:
        if get_batch(url) != None:
            for batch, total in get_batch(url):
                for line in tqdm(batch.json()['results']):
                    data.append(line)
                    
            ec_num_str = '_'.join(ec_num.split('.'))
            reviewed_num = 0 if reviewed=='false' else 1
            file_name = f'{enzyme_type}_ec{ec_num_str}_reviewed{reviewed_num}_{len(df)}.csv'  if file_name=='' else file_name
            save_path += file_name
            if download and len(data)>0:
                with open(save_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)
                print(f'Save file for ec:{ec_num} with reviewed={reviewed} to path: {save_path}')
        
    return data
        




def collate_target_enzyme():

    def contains_english_chars(input_str):
        # Regular expression to match English letters
        pattern = re.compile('[a-zA-Z]')
        
        # Check if the input string contains English letters
        return bool(pattern.search(input_str))
   

    all_data = pd.read_csv('data/ecnumber_brenda.csv')
    if all_data.empty == False:

        for index, row in all_data.iterrows():    
            for enzyme in ENZYME_TYPES:
                if enzyme in row['Recommended Name'].lower():
                    if contains_english_chars(row['EC Number']) == False:                                           
                        crawl_uniprot_by_ecnumber(ec_num=row['EC Number'], enzyme_type=enzyme,reviewed='true',save_path=f'data/brenda/{enzyme}/reviewed/')
                        crawl_uniprot_by_ecnumber(ec_num=row['EC Number'], enzyme_type=enzyme,reviewed='false',save_path=f'data/brenda/{enzyme}/unreviewed/')
    else:
        print("There's no ec number data to refer for digestive enzyme!")




def crawl_uniprot_by_query(query, download=True,
    save_path = '', file_name='',
    fields = ['ec','accession','id', 'protein_name', 'gene_names', 'organism_name', 'length', 'cc_function', 'cc_biotechnology','cc_interaction'],
    headers = ['EC number', 'Entry', 'Entry Name', 'Protein names', 'Gene Names', 'Organism', 'Length', 'Function [CC]', 'Biotechnological use', 'Interacts with'],
    return_format = 'tsv',
    reviewed = 'true'): 

    url = f'https://rest.uniprot.org/uniprotkb/search?fields={",".join(fields)}&format={return_format}&query=({query})%20AND%20(reviewed%3A{reviewed})&size=500'
    
    data = []
    
    
    if(return_format == 'tsv' ):
    
        if get_batch(url) != None:
            for batch, total in get_batch(url):
                for line in batch.text.splitlines()[1:]:
                    data.append(line.split('\t'))       # LUN 2024/07/30 check if there's blank in string    
            df = pd.DataFrame(data, columns=headers)
                
            reviewed_num = 0 if reviewed=='false' else 1
            file_name = f'{query}_reviewed{reviewed_num}_{len(df)}.csv'  if file_name=='' else file_name
            save_path = os.path.join(save_path, file_name)
            if download and len(data)>0: 
                df.to_csv(save_path, index=False)
                # print(f'Save file for query:{query} with reviewed={reviewed} to path: {save_path}')
    else:

        if get_batch(url) != None:
            for batch, total in get_batch(url):
                for line in batch.json()['results']:
                    data.append(line)
                    
            reviewed_num = 0 if reviewed=='false' else 1
            file_name = f'{query}_reviewed{reviewed_num}_{len(data)}.json'  if file_name=='' else file_name
            save_path = os.path.join(save_path, file_name)
            if download and len(data)>0: 
                with open(save_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)    
                # print(f'Save file for query:{query} with reviewed={reviewed} to path: {save_path}')

    return data


def download_fasta(query, save_path='', file_name='', compressed='false', reviewed='true',):
    url = f'https://rest.uniprot.org/uniprotkb/stream?compressed={compressed}&format=fasta&query=({query})%20AND%20(reviewed%3A{reviewed})'
        
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch data. HTTP Status Code: {response.status_code}")


    all_fastas = response.text
    fasta_list = re.split(r'\n(?=>)', all_fastas)
            
    reviewed_num = 0 if reviewed=='false' else 1
    file_name = f'{query}_reviewed{reviewed_num}_{len(fasta_list)}.fasta'  if file_name=='' else file_name
    save_path = os.path.join(save_path, file_name)

    def save_to_fasta(sequences, filename):
        with open(filename, 'w') as f:
            for seq in sequences:
                f.write(f'{seq}\n')
        
        
        # print(f'Save .fasta file for query:{query} with reviewed={reviewed} to path: {save_path}')


    save_to_fasta(fasta_list, save_path)

    
    



if __name__ == '__main__':
    for enzyme_type in ENZYME_TYPES:
        crawl_uniprot_by_query(query=enzyme_type, reviewed='true',save_path='data/uniprot/reviewed/')

    # the amount of unreviewed data is too large! 
    for enzyme_type in ENZYME_TYPES:
        crawl_uniprot_by_query(query=enzyme_type, reviewed='false',save_path='data/uniprot/unreviewed/')


