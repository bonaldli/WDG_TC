# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:32:39 2019

@author: zlibn
"""
import requests
class MTRobot:
    def __init__(self):
        """
        Explanation:
        -----------
        bot_token: MTRobot token
        bot_chatID: receiver's ID
        msg: message
        """
        
    def sendtext(bot_message):
        bot_token = '909014457:AAGtI8Qeh99i4_R5EkbMb1NEAGhpdbZ0lY0'
        bot_chatID = '759131145'
        send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        
        return None
   
