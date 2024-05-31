## Task4 基于大模型的语音交互智能体助手设计

  在百度的AI原生应用开发平台https://appbuilder.cloud.baidu.com/上创建一个应用，将个人密钥和应用ID填入，运行程序。可实现应用的语音交互。

<img src="C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240601024021965.png" alt="image-20240601024021965" style="zoom: 67%;" />     <img src="C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240601023121689.png" alt="image-20240601023121689" style="zoom: 67%;" />      

智能对话机器人的核心工作流程可以分为以下几个步骤：

1. 语音输入：

用户通过语音向机器人发出指令或提问。这是整个交互过程的起点。

2. ASR（Automatic Speech Recognition，自动语音识别）：

语音输入首先经过ASR模块处理。ASR的作用是将用户的语音转换为可理解的文本。这一步骤对于后续处理至关重要，因为只有将语音准确地转化为文本，机器人才能正确理解用户的意图。

3. 智能体Agent：

转换后的文本会被传递到智能体Agent进行处理。智能体Agent是整个系统的核心，它负责理解用户的需求，并调用相应的功能模块来生成响应。例如，如果用户询问天气情况，智能体Agent会调用天气查询工具来获取相关信息。

4. TTS（Text-to-Speech，文本转语音）：

智能体Agent生成响应后，会将结果传递给TTS模块。TTS模块负责将文本转换为自然流畅的语音，并反馈给用户。这样，用户就可以听到机器人的回答了。

![image-20240601023827814](C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240601023827814.png)