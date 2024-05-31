import requests
from bs4 import BeautifulSoup

# 假设网站结构是每卷一个页面，通过遍历卷号来爬取
base_url = "https://www.diyifanwen.com/guoxue/quantangshi/" 

volume_urls = []  
headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
} 

# 获取所有卷链接
def crawl_pagination(base_url, start_page, end_page):
    for i in range(start_page, end_page + 1):
        page_url = base_url
        if i != 1:
            page_url = f"{base_url}index_{i}.html"
        response = requests.get(page_url,headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            lis = soup.find_all('li')
            for li in lis:
                a_tag = li.find('a', href=True)
                if a_tag:
                    volume_link = a_tag['href'][2:]
                    volume_urls.append(volume_link)  


crawl_pagination(base_url, 1, 20)

# 爬取卷内的所有内容
def crawe_rolltext(num, roll_url):
    all_text = ""
    if not roll_url.startswith("http://") and not roll_url.startswith("https://"):
            roll_url = "http://" + roll_url
    response = requests.get(roll_url, headers=headers)
    
    if response.status_code == 200:
        for i in range(1,10):
            content_url = roll_url
            if i != 1:
                content_url = content_url.replace(".htm", f"_{i}.htm")
            response = requests.get(content_url, headers=headers)
            response.encoding = 'gbk'
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, 'html.parser')
            content_element = soup.find('div', class_='content')
            if content_element:
                if num == 1:
                    all_paragraphs_text = content_element.get_text(strip=True)
                else:
                    paragraphs = content_element.find_all('p')
                    all_paragraphs_text = "\n".join([para.get_text() for para in paragraphs])
                # content_text = content_element.get_text(strip=True)
                all_text += all_paragraphs_text + "\n"
                
    
    with open(f"./roll_{num}.txt",'w', encoding='utf-8') as file:
        file.write(all_text)  

# print(len(volume_urls))

for index, volume_url in enumerate(volume_urls):
    crawe_rolltext(index+1, volume_url)
      