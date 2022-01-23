# Human-Activity-Recognition-using-Smartphone

# Data 
30 participants performed activities of daily living while carrying a waist-mounted smartphone. The phone was configured to record two implemented sensors (accelerometer and gyroscope). For these time series the directors of the underlying study performed feature generation and generated the dataset by moving a fixed-width window of 2.56s over the series. 

# Labels Distributed 

![newplot (5)](https://user-images.githubusercontent.com/85125898/150678362-bc2b65e2-44f2-4b82-9e8b-270d1a66534e.png)

Although there are fluctuations in the label counts, the labels are quite equally distributed.

Assuming the participants had to walk the same number of stairs upwards as well as downwards and knowing the smartphones had a constant sampling rate, there should be the same amount of datapoints for walking upstairs and downstairs.
Disregarding the possibility of flawed data, the participants seem to walk roughly 10% faster downwards.

# Activity Exploration 

𝐀𝐫𝐞 𝐓𝐡𝐞 𝐀𝐜𝐭𝐢𝐯𝐢𝐭𝐢𝐞𝐬 𝐒𝐞𝐩𝐚𝐫𝐚𝐛𝐥𝐞?

The dataset is geared towards classifying the activity of the participant. Let us investigate the separability of the classes.

![__results___11_0](https://user-images.githubusercontent.com/85125898/150678462-d2074207-0cf1-43c5-9d5d-432b46a6f8e3.png)

In plot 1 you can clearly see the activities are mostly separable.

Plot 2 reveals personal information of the participants. Everybody has for example an unique/sparable walking style (on the upper right). Therefore the smartphone should be able to detect what you are doing and also who is using the smartphone (if you are moving around with it).

𝐇𝐨𝐰 𝐆𝐨𝐨𝐝 𝐀𝐫𝐞 𝐓𝐡𝐞 𝐀𝐜𝐭𝐢𝐯𝐢𝐭𝐢𝐞𝐬 𝐒𝐞𝐩𝐚𝐫𝐚𝐛𝐥𝐞?

Without much preprocessing and parameter tuning a simple LGBMClassifier should work decently.

𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲 𝐨𝐧 𝐭𝐞𝐬𝐭𝐬𝐞𝐭:	𝟎.𝟗𝟓𝟓𝟕
With a basic untuned model the activity of the smartphone user can be predicted with an 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲 𝐨𝐟 𝟗𝟓%.
This is pretty striking regarding six equally distributed labels.

# Participant Exploration

𝐇𝐨𝐰 𝐆𝐨𝐨𝐝 𝐀𝐫𝐞 𝐭𝐡𝐞 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭𝐬 𝐒𝐞𝐩𝐚𝐫𝐚𝐛𝐥𝐞?

As we have seen in the second t-SNE plot the separability of the participants seem to vary regarding their activity. Let us investigate this a little bit by fitting the same basic model to the data of each activity separately.

![Human Activity Recognition using Smartphone Data with Machine Learning - Jupyter Notebook - Google Chrome 1_23_2022 6_03_38 PM](https://user-images.githubusercontent.com/85125898/150678631-7b491270-9ad3-40f7-9633-5acc78a0f814.png)

Detecting the correct participant regarind their current activity is not alone possible but astonishing accurate regarding the 30 different persons (94% by walking style).

