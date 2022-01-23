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

𝐃𝐞𝐭𝐞𝐜𝐭𝐢𝐧𝐠 𝐭𝐡𝐞 𝐜𝐨𝐫𝐫𝐞𝐜𝐭 𝐩𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭 regarind their current activity is not alone possible but 𝐚𝐬𝐭𝐨𝐧𝐢𝐬𝐡𝐢𝐧𝐠 𝐚𝐜𝐜𝐮𝐫𝐚𝐭𝐞 regarding the 30 different persons (𝟗𝟒% 𝐛𝐲 𝐰𝐚𝐥𝐤𝐢𝐧𝐠 𝐬𝐭𝐲𝐥𝐞).


𝐖𝐡𝐢𝐜𝐡 𝐒𝐞𝐧𝐬𝐨𝐫 𝐈𝐬 𝐌𝐨𝐫𝐞 𝐈𝐦𝐩𝐨𝐫𝐭𝐚𝐧𝐭 𝐅𝐨𝐫 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐲𝐢𝐧𝐠 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭𝐬 𝐁𝐲 𝐖𝐚𝐥𝐤𝐢𝐧𝐠 𝐒𝐭𝐲𝐥𝐞?

![__results___19_0](https://user-images.githubusercontent.com/85125898/150678729-345b5da7-fe1f-4ab1-9352-9671806e3afe.png)

The accelerometer supplies slightly more information. Both sensors are important for classification and refraining from using both sensors will be a drawback for the quality of the model.


𝐇𝐨𝐰 𝐋𝐨𝐧𝐠 𝐃𝐨𝐞𝐬 𝐓𝐡𝐞 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭 𝐔𝐬𝐞 𝐓𝐡𝐞 𝐒𝐭𝐚𝐢𝐫𝐜𝐚𝐬𝐞?

![__results___21_0](https://user-images.githubusercontent.com/85125898/150678777-8de22707-b964-4fd9-bd9c-e59e0e5707dd.png)

Nearly all participants have more data for walking upstairs than downstairs. Assuming an equal number of up- and down-walks the 𝐩𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭𝐬 𝐧𝐞𝐞𝐝 𝐥𝐨𝐧𝐠𝐞𝐫 𝐰𝐚𝐥𝐤𝐢𝐧𝐠 𝐮𝐩𝐬𝐭𝐚𝐢𝐫𝐬.
Furthermore the range of the duration is narrow and adjusted to the conditions. A young person being ~50% fast in walking upstairs than an older one is reasonable.


𝐇𝐨𝐰 𝐌𝐮𝐜𝐡 𝐃𝐨𝐞𝐬 𝐓𝐡𝐞 𝐔𝐩-/𝐃𝐨𝐰𝐧𝐬𝐭𝐚𝐢𝐫𝐬 𝐑𝐚𝐭𝐢𝐨 𝐕𝐚𝐫𝐲?

![__results___23_0](https://user-images.githubusercontent.com/85125898/150678835-f5ac4e14-4ac4-4330-9b08-f2262d3f62f9.png)

There is a wide range in between the participants for their 𝐫𝐚𝐭𝐢𝐨 𝐨𝐟 𝐮𝐩-/𝐝𝐨𝐰𝐧-𝐰𝐚𝐥𝐤𝐢𝐧𝐠. Since this represents their physical condition I can imagine a 𝐜𝐨𝐫𝐫𝐞𝐥𝐚𝐭𝐢𝐨𝐧 𝐭𝐨 𝐭𝐡𝐞𝐢𝐫 𝐚𝐠𝐞 𝐚𝐧𝐝 𝐡𝐞𝐚𝐥𝐭𝐡 (𝐬𝐩𝐞𝐜𝐮𝐥𝐚𝐭𝐢𝐯𝐞).


𝐀𝐫𝐞 𝐓𝐡𝐞𝐫𝐞 𝐂𝐨𝐧𝐬𝐩𝐢𝐜𝐮𝐢𝐭𝐢𝐞𝐬 𝐈𝐧 𝐓𝐡𝐞 𝐒𝐭𝐚𝐢𝐫𝐜𝐚𝐬𝐞 𝐖𝐚𝐥𝐤𝐢𝐧𝐠 𝐃𝐮𝐫𝐚𝐭𝐢𝐨𝐧 𝐃𝐢𝐬𝐭𝐫𝐢𝐛𝐮𝐭𝐢𝐨𝐧?

![__results___25_0](https://user-images.githubusercontent.com/85125898/150678900-fbd3d254-1199-4145-addc-49bf2b31abfd.png)

As aspected from most real world data the duration walking on the staircase is normally distributed


𝐈𝐬 𝐓𝐡𝐞𝐫𝐞 𝐀 𝐔𝐧𝐢𝐪𝐮𝐞 𝐖𝐚𝐥𝐤𝐢𝐧𝐠 𝐒𝐭𝐲𝐥𝐞 𝐅𝐨𝐫 𝐄𝐚𝐜𝐡 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭?

![__results___27_0](https://user-images.githubusercontent.com/85125898/150678927-8f383cec-3613-41d2-8ffb-137c1e6585e6.png)


𝐇𝐨𝐰 𝐋𝐨𝐧𝐠 𝐃𝐨𝐞𝐬 𝐓𝐡𝐞 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭 𝐖𝐚𝐥𝐤?

![__results___29_0](https://user-images.githubusercontent.com/85125898/150678952-4ec3af61-0f7a-437e-83c5-2de98d4a05bc.png)

Since the duration of each participant walking is distributed over a range I assume the participants had a fixed walking distance for their experiment rather than a fixed duration.


𝐈𝐬 𝐓𝐡𝐞𝐫𝐞 𝐀 𝐔𝐧𝐢𝐪𝐮𝐞 𝐒𝐭𝐚𝐢𝐫𝐜𝐚𝐬𝐞 𝐖𝐚𝐥𝐤𝐢𝐧𝐠 𝐒𝐭𝐲𝐥𝐞 𝐅𝐨𝐫 𝐄𝐚𝐜𝐡 𝐏𝐚𝐫𝐭𝐢𝐜𝐢𝐩𝐚𝐧𝐭?

![__results___31_0](https://user-images.githubusercontent.com/85125898/150678976-5962657f-bd62-466a-a6f3-567109e8fda8.png)

In most of the plots a structure with 𝐭𝐰𝐨 𝐜𝐥𝐮𝐬𝐭𝐞𝐫𝐬 is again recognizable. Going 𝐮𝐩 𝐚𝐧𝐝 𝐝𝐨𝐰𝐧 𝐭𝐡𝐞 𝐬𝐭𝐚𝐢𝐫𝐬 𝐟𝐨𝐫 𝐭𝐰𝐨 𝐭𝐢𝐦𝐞𝐬 is likely for the experiment.
I will review the durations for this assumption with a small experiment on my stairs.


# SVM for Multiclass Classification 

![Human Activity Recognition using Smartphone Data with Machine Learning - Jupyter Notebook - Google Chrome 1_23_2022 6_19_32 PM](https://user-images.githubusercontent.com/85125898/150679174-37c95ecd-e04b-47af-967a-191b06a7b46a.png)

![download](https://user-images.githubusercontent.com/85125898/150679213-bc8f927e-6749-42fc-90a8-cc9d2b097419.png)

