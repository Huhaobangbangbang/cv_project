"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/11/4 8:31 AM
"""

#coding:utf-8
import requests
from bs4 import BeautifulSoup

def get_all_websites():
    url = "https://www.baidu.com"
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "html.parser")
    # find_all会将所有满足条件的值取出，组成一个list
    link_nodes = soup.find_all("a")
    for node in link_nodes:
        print(node.get("href"))
get_all_websites()