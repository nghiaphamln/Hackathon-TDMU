from bs4 import BeautifulSoup
import requests
import csv

url = 'http://quotes.toscrape.com'


def tacgiaLink(link):
    respone = requests.get(link)
    soup = BeautifulSoup(respone.content, 'html5lib')
    return soup.find('div', class_='author-details').span.text

with open('Quote.csv', 'w', newline='') as file:
    fieldnames = ['Tacgia', 'Link', "Namsinh", "Quote"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for i in range(1, 6):
        respone = requests.get(url)
        soup = BeautifulSoup(respone.content, 'html5lib')
        for x in soup.find_all('div', class_='quote'):
            result = x.span.text
            author = x.small.text
            link = url + x.a['href']
            date_of_birth = tacgiaLink(link)
            print(f"Tacgia: {author}")
            print(f"Link: {link}")
            print(f"Ngaysinh: {date_of_birth}")
            print(f"Quote: {result}\n")


            writer.writerow({"Tacgia": author, "Link": link, "Namsinh": date_of_birth, "Quote": result})

    file.close()
    