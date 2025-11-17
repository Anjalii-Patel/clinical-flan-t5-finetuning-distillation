import pandas as pd
import random


conversation_1 = """
How are you feeling today? Tell me about your main concerns.
I’ve been feeling very tired lately, and my muscles ache more than usual. Even when I sleep for 8 hours, I wake up feeling exhausted.
Have you experienced this fatigue before, or is it recent?
I felt something similar last year, but it went away after a few weeks. This time, it’s been going on for over a month.
Have you noticed any other symptoms, such as changes in weight, appetite, or energy levels?
Yes, I’ve lost a little weight unintentionally, and my appetite has decreased. I also feel short of breath sometimes when climbing stairs.
Have you ever been hospitalized or had surgery for any conditions?
I had my appendix removed about five years ago, but nothing since then.
Do you have any known allergies or reactions to medications?
Not that I’m aware of.
What’s your daily routine like? Are you getting proper nutrition and staying active?
I work long hours at a desk job and don’t exercise much. My meals are often quick snacks, and I probably don’t eat as many vegetables as I should.
Does anyone in your family have similar health concerns or conditions?
My father has diabetes, and my mother had anemia when she was younger.
"""

conversation_2 = """
What brings you in today?
I’ve been experiencing irregular menstrual cycles for the past six months. Sometimes I skip a month, and other times the bleeding lasts much longer than normal.
Have you ever had issues with your menstrual cycle in the past?
Yes, when I was younger, my cycles were always irregular, but they stabilized in my mid-twenties.
Are there any other symptoms you’ve noticed, such as abdominal pain, changes in mood, or excessive fatigue?
I do feel very tired lately, and I’ve also gained some weight, especially around my midsection. My mood has been a bit off too—sometimes I feel more anxious than usual.
Have you had any hospitalizations or surgeries?
I haven’t had any surgeries, but I was hospitalized once for a severe flu three years ago.
Do you take any medications or have any allergies?
I’m not on any medications, and I don’t have any known allergies.
How would you describe your lifestyle and activity levels?
I lead a pretty sedentary life since I work from home. I don’t get much exercise, and my diet is inconsistent.
Does your family have a history of gynecological or hormonal conditions?
Yes, my mother had issues with fibroids, and my sister was recently diagnosed with polycystic ovary syndrome (PCOS).
"""

conversation_3 = """
Hello, it’s good to see you today. What can I help you with?
I’ve been feeling really tired for the past couple of months, no matter how much I sleep.
How long has this been happening?
It started gradually about two months ago. Now, I feel exhausted all the time.
Have you noticed any changes in your appetite, weight, or sleep patterns?
My appetite has decreased, and I’ve lost a few pounds, but my sleep has been fine.
Do you have any history of thyroid problems or other health issues?
I was diagnosed with hypothyroidism a few years ago, but I’ve been on medication for it.
What’s your daily routine like?
I work a desk job, so I’m sitting most of the day, and I don’t get much exercise. I’ve also been feeling a lot of stress from work.
Are you currently taking any medications?
Yes, I take levothyroxine for my thyroid, and I don’t have any allergies.
Does anyone in your family have thyroid or other health issues?
Yes, my mother has thyroid problems, and my father has high blood pressure.
"""

conversation_4 = """
Hi there, how are you feeling today? What brings you in?
I’ve been dealing with acne, mostly around my chin and jawline. It’s been pretty persistent.
How long has this been going on?
It started about a month ago, and it’s been getting worse.
Have you tried any new skin products or made changes to your routine?
I started using a new moisturizer, and I’ve been wearing makeup more frequently.
Is there any other skin concern, like dryness or redness?
No redness, but my skin does feel a bit drier than usual.
Have you had acne issues before?
I had some acne as a teenager, but it cleared up when I was in my twenties.
How would you describe your lifestyle?
I have a stressful job, and I tend to touch my face a lot while I work.
Are you on any medications right now?
No, I’m not on any medications, and I don’t have any allergies.
"""

conversation_5 = """
Good day! What brings you in today?
I’ve been experiencing tightness in my chest and shortness of breath. It’s been happening quite often.
How long have you been having these symptoms?
It started about a week ago, and it's been occurring more frequently recently.
Do you have any chest pain, dizziness, or swelling in your legs?
No chest pain, but I do feel lightheaded sometimes, and I’ve noticed some swelling in my ankles.
Do you have any history of lung or heart conditions?
I have asthma, but it’s been controlled for years. I haven’t had any issues with my chest for a long time.
Are you currently on any medications?
I just use an inhaler when needed.
How would you describe your lifestyle? Do you smoke or get enough exercise?
I used to smoke but quit five years ago. I don’t exercise much anymore.
Does anyone in your family have heart or lung conditions?
My father had a heart attack in his forties, and my grandmother had COPD.
"""

conversation_6 = """
Hello! What can I do for you today?
I’ve noticed a lump in my neck, and I’m concerned about it.
When did you first notice the lump?
It’s been there for about two weeks now, but it hasn’t grown in size.
Does it hurt or feel tender?
It’s firm to the touch, but not painful unless I press on it.
Have you had any illnesses, like a sore throat, recently?
I had a mild sore throat last month, but it cleared up quickly.
Do you have any history of thyroid issues or other lymph node problems?
I’ve had my thyroid checked before, and it was normal. No issues with lymph nodes either.
Are you on any medications or have any allergies?
I’m not on any medications and don’t have any allergies.
Does anyone in your family have thyroid problems or cancer?
My aunt had thyroid cancer, but she’s been in remission for several years now.
"""

conversation_7 = """
Hi there! What seems to be the problem today?
I’ve been having lower back pain for the past month. It’s especially bad after I sit for long periods.
When did you first notice the pain?
It started about a month ago. The pain has been gradually increasing.
Do you experience any numbness or tingling in your legs?
No, just stiffness in my lower back when I stand up after sitting.
Have you had any injuries or accidents that may have affected your back?
I haven’t had any major injuries, but I’ve lifted heavy objects in the past, which may have strained my back.
What’s your daily activity level like?
I work at a desk all day, so I’m sitting most of the time. I don’t exercise much.
Are you on any medications for the pain?
I take ibuprofen occasionally when it’s really bad.
Does anyone in your family have back problems or arthritis?
My mother has some back issues, and my grandfather had osteoarthritis.
"""

conversation_8 = """
Good morning! What brings you in today?
I’ve been having frequent urination, especially at night. It’s been really bothering me.
How long has this been happening?
It started about two weeks ago, and it’s been getting worse. I wake up three to four times a night to go to the bathroom.
Have you had any pain or burning sensation while urinating?
No, there’s no pain, just the frequent urge to go. It’s mostly just annoying.
Do you have a history of urinary tract infections or kidney problems?
I haven’t had any UTIs in the past, and I don’t have any history of kidney issues.
Are you on any medications?
I’m not currently on any medications.
How’s your fluid intake and lifestyle? Do you drink a lot of caffeine or alcohol?
I drink a lot of coffee during the day, and I’ve been drinking a bit more alcohol recently, but I don’t really think that’s the cause.
Has anyone in your family had issues with urinary problems or diabetes?
My mother has diabetes, but she doesn’t have any urinary problems. My father is healthy.
"""

conversation_9 = """
Hello, what can I do for you today?
I’ve been feeling a lot of joint pain in my knees and elbows lately. It’s been making it hard to move around.
When did you first notice the pain?
It started about a month ago, and it’s been gradually getting worse, especially after I’ve been sitting for a while.
Have you noticed any swelling, redness, or warmth around the joints?
Yes, I’ve noticed some swelling in my knees, but not much redness or warmth.
Do you have a history of any joint-related conditions, such as arthritis?
I don’t have a history of arthritis, but my mother has rheumatoid arthritis, and my grandfather had osteoarthritis.
Are you on any pain medication or treatments?
I’ve been taking over-the-counter pain relievers like ibuprofen, but they only help a little.
How would you describe your daily activity level? Do you do any physical exercise?
I’m not very active, to be honest. I mostly sit at work and don’t do much physical activity outside of that.
Has your family had any history of autoimmune diseases or joint problems?
Yes, like I mentioned, my mother has rheumatoid arthritis.
"""

conversation_10 = """
Hi there! How can I help you today?
I’ve been feeling a lot of chest discomfort, and it sometimes feels like there’s pressure in my chest.
When did this start?
It began about a week ago, and I’ve noticed it happening mostly after eating or when I’m stressed.
Do you have any shortness of breath, dizziness, or pain radiating to your arms?
No shortness of breath or pain radiating, but I do feel a little lightheaded when it happens.
Have you had any history of heartburn or digestive issues?
Yes, I’ve had heartburn in the past, and sometimes it feels like the discomfort is related to that.
Are you on any medications?
I take an over-the-counter antacid when the heartburn gets bad, but that’s about it.
Do you have any history of heart disease in your family?
My father had heart disease and had a heart attack about five years ago, but he’s doing fine now.
Have you noticed any patterns in your diet or lifestyle that might be triggering the discomfort?
I’ve been eating a lot of spicy food lately, and I tend to skip meals during busy days. I also drink a lot of coffee.
"""

conversation_11 = """
Good afternoon! How are you feeling today?
I’ve been experiencing sharp abdominal pain, especially on the right side.
How long have you had this pain?
It started about three days ago, and it’s been getting worse. It’s very sharp at times, and it’s uncomfortable when I move.
Have you noticed any other symptoms, such as nausea, vomiting, or fever?
I haven’t had nausea or vomiting, but I’ve had a mild fever and some bloating.
Do you have any history of gastrointestinal issues, such as ulcers or appendicitis?
No, I’ve never had any gastrointestinal issues in the past. I don’t think I’ve had appendicitis either.
Are you on any medications or have any allergies?
I’m not on any regular medications, and I don’t have any known allergies.
What’s your diet like? Have you had any changes in your eating habits recently?
I’ve been eating a bit more fast food lately due to a busy work schedule.
Does anyone in your family have a history of gastrointestinal or abdominal issues?
No, no one in my family has had issues with their digestive system.
"""

conversation_12 = """
Hello! What brings you in today?
I’ve been dealing with severe headaches for the past week. They tend to come in the evenings.
How long have you been experiencing the headaches?
It started about a week ago, and they’ve been happening daily since.
Do you have any other symptoms, like dizziness, nausea, or sensitivity to light?
Yes, I do feel lightheaded sometimes, and bright lights make the headache worse.
Have you had any similar headaches in the past?
I used to get headaches when I was stressed, but these feel different. They’re much more intense.
Are you on any medications right now?
I take an occasional ibuprofen for the pain, but nothing regular.
Do you have a history of migraines or other neurological issues?
No, I don’t have a history of migraines or any neurological problems.
How is your sleep and stress level?
I haven’t been sleeping well, and I’ve been feeling very stressed at work.
"""

conversation_13 = """
Hi, good to see you. What’s troubling you today?
I’ve been having a lot of back pain recently. It’s mostly in my lower back.
How long has this been bothering you?
The pain started a few weeks ago, but it’s been getting worse over time.
Do you experience any numbness, tingling, or weakness in your legs?
No, it’s just pain in my back, especially after I sit for a while or bend over.
Have you had any injuries, accidents, or a history of back problems?
I’ve had some mild back issues in the past, but nothing serious. No major injuries.
What’s your daily activity level like? Do you sit a lot or engage in physical activity?
I have a desk job, so I’m sitting most of the day. I don’t get much exercise.
Are you taking any medications for the pain?
I’ve been taking over-the-counter pain relievers like acetaminophen, but it doesn’t seem to help much.
Does anyone in your family have a history of back pain or spinal conditions?
Yes, my mother has chronic back pain and has been dealing with it for many years.
"""

conversation_14 = """
Good day! How can I assist you today?
I’ve been feeling anxious a lot lately, and it’s affecting my daily routine. I’m having trouble concentrating and sleeping.
How long have you been feeling this way?
It’s been going on for about a month now. It started slowly but has progressively gotten worse.
Do you experience physical symptoms, such as rapid heartbeat, sweating, or dizziness?
Yes, I feel my heart racing sometimes, especially when I’m stressed. I also sweat more than usual.
Have you had any past issues with anxiety or depression?
I’ve had some anxiety in the past, but this feels different. It’s more intense now.
Are you taking any medications for your anxiety or mood?
No, I’m not on any medications for anxiety or depression.
How would you describe your daily routine and lifestyle?
I have a busy schedule, with work and family responsibilities, but I don’t get enough exercise or downtime.
Does anyone in your family have a history of mental health issues?
Yes, my father has dealt with anxiety and depression, and my sister was diagnosed with generalized anxiety disorder (GAD).
"""

conversation_15 = """
Hello! What brings you in today?
I’ve been dealing with irregular periods for the past couple of months. Sometimes they’re late, and other times they’re much heavier than usual.
How long has this been happening?
It started about two months ago. My periods are all over the place now.
Have you noticed any other symptoms, such as changes in mood, skin, or weight gain?
Yes, I’ve gained some weight, and I feel more emotional and anxious. My skin has also become more oily.
Have you had irregular periods in the past?
I’ve always had regular periods, but this has been a new issue.
Do you have any history of hormonal disorders, such as polycystic ovary syndrome (PCOS)?
No history of PCOS, but my mother had fibroids and went through early menopause.
Are you on any medications right now?
I’m not on any medications.
How’s your lifestyle? Do you eat a balanced diet or exercise regularly?
I have a sedentary lifestyle, and my diet isn’t great. I’ve been eating a lot of fast food recently.
"""

conversation_16 = """
Hello, how are you today?
I’ve been feeling really tired and lethargic for the past couple of weeks. No matter how much I rest, I feel drained.
When did you first notice these symptoms?
It started about two weeks ago, and it’s been a constant struggle to get through the day.
Have you had any other symptoms, like fever, sore throat, or body aches?
No, I haven’t had any other symptoms, just the fatigue. I’m not sure what’s causing it.
Do you have any history of anemia or thyroid problems?
I’ve had anemia in the past, but my iron levels have been normal recently. No thyroid issues.
Are you taking any medications or vitamins?
I take a multivitamin, but that’s all.
How would you describe your daily routine? Do you have a lot of stress or physical activity?
I have a fairly busy job, but I don’t exercise much. I’m constantly working and feel mentally drained.
Does anyone in your family have a history of thyroid issues or chronic fatigue?
No history of thyroid problems, but my sister has chronic fatigue syndrome.
"""

conversation_17 = """
Hi there! What’s bothering you today?
I’ve been having frequent nosebleeds, especially when I wake up in the morning.
How long have you been experiencing this?
It started about a month ago. It happens almost every morning when I wake up.
Have you noticed any other symptoms, such as headaches or sinus congestion?
I don’t have headaches or congestion, just the nosebleeds.
Do you have a history of allergies, sinus issues, or bleeding disorders?
I have mild allergies, but no bleeding disorders or major sinus issues.
Are you on any medications right now?
I’m on an antihistamine for my allergies, but that’s about it.
How’s the climate where you live? Do you think it could be contributing to the nosebleeds?
It’s pretty dry here, especially in the winter, so I think that might be the cause.
Does anyone in your family have a history of frequent nosebleeds?
No, I’m the only one in my family who has them.
"""

conversation_18 = """
Good morning! What can I help you with today?
I’ve been having pain in my right shoulder and neck for the past few days. It’s been getting worse and affecting my movement.
When did this pain start?
It started about four days ago, and it’s been progressively worse. I feel stiff and have limited movement.
Have you had any injuries or accidents that may have caused this pain?
I didn’t have any major accidents, but I’ve been lifting heavy boxes at work lately.
Do you have a history of neck or shoulder problems?
No, I’ve never had issues with my neck or shoulders before.
Are you taking any medications for the pain?
I’ve been using some over-the-counter pain relief creams, but they haven’t helped much.
How’s your lifestyle? Do you sit for long hours or engage in physical activity?
I work at a desk all day, so I sit for long periods, but I don’t exercise much.
Does anyone in your family have a history of musculoskeletal issues?
No, there’s no family history of musculoskeletal problems.
"""

conversation_19 = """
Hi! How are you doing today?
I’ve been experiencing some dizziness and lightheadedness, especially when I stand up quickly.
How long has this been happening?
It started about a week ago, and it’s been happening on and off since then.
Do you have any other symptoms, such as headaches, nausea, or chest pain?
No, just dizziness. I haven’t had headaches or chest pain, just the feeling of fainting when I stand too fast.
Do you have a history of heart problems or low blood pressure?
I have low blood pressure, but it’s been well controlled.
Are you on any medications for your blood pressure?
I take medication for low blood pressure, but it’s been stable.
How’s your diet and hydration? Do you drink enough fluids?
I tend to forget to drink water throughout the day, so I probably don’t drink enough.
Does anyone in your family have a history of low blood pressure or heart disease?
My father has low blood pressure, but no one else in my family has heart disease.
"""

conversation_20 = """
Hello! What’s troubling you today?
I’ve been feeling pain in my knees and hips, especially after I walk or stand for a while.
How long have you been experiencing this pain?
It started about two weeks ago, and it’s been getting progressively worse.
Have you noticed any swelling or redness in the joints?
Yes, my knees feel swollen, and they sometimes get a little red.
Do you have a history of arthritis or other joint-related conditions?
My mother had osteoarthritis, but I’ve never been diagnosed with it myself.
Are you taking any medications for the pain?
I’ve been using ibuprofen occasionally, but it doesn’t seem to help much.
How’s your physical activity? Do you get a lot of exercise or sit for long periods?
I don’t exercise much, and I have a desk job, so I sit a lot.
Does anyone in your family have a history of joint issues or arthritis?
As I mentioned, my mother had osteoarthritis.
"""

conversation_21 = """
Good afternoon! What brings you in today?
I’ve been experiencing a lot of hair loss recently, and I’m really concerned about it.
How long has this been happening?
It started about a month ago, and I’ve noticed a significant amount of hair shedding every day.
Have you had any changes in your diet, stress levels, or medications recently?
I’ve been under a lot of stress at work, and I haven’t been eating very well. No changes in medications.
Do you have any history of thyroid problems or other hormonal imbalances?
I’ve had thyroid problems in the past, and I’ve been on medication for that. I’m also concerned it could be hormonal.
Are you on any medications right now?
I take medication for my thyroid, but no other medications.
How would you describe your lifestyle and stress levels?
I’m often stressed, especially with work, and I don’t get enough sleep or exercise.
Does anyone in your family have a history of hair loss or thyroid issues?
Yes, my mother has thyroid issues and has also experienced hair loss.
"""

conversation_22 = """
Hi! How are you today?
I’ve been experiencing pain and a burning sensation when urinating.
When did this start?
It started about a week ago, and it’s been bothering me daily.
Do you have any other symptoms, like fever, lower abdominal pain, or blood in your urine?
I haven’t noticed any blood, but I have had some mild lower abdominal discomfort and a feeling of pressure.
Have you had urinary tract infections (UTIs) in the past?
Yes, I’ve had a few UTIs in the past, but this feels different.
Are you on any medications for your symptoms?
I haven’t started any medications yet, but I’m thinking of getting something for the pain.
Do you have a history of kidney stones or other urinary issues?
No, I’ve never had kidney stones, but I’m concerned this might turn into something more serious.
How’s your hydration? Do you drink enough water throughout the day?
I don’t drink as much water as I should, especially with my busy schedule.
"""

conversation_23 = """
Hello, what’s going on today?
I’ve been feeling really short of breath lately, especially when I’m doing any physical activity.
How long have you been having this issue?
It started about a week ago, and it’s been progressively getting worse.
Have you noticed any chest pain, dizziness, or coughing?
I haven’t had any chest pain or coughing, but I do feel lightheaded sometimes.
Do you have a history of asthma, allergies, or heart disease?
I have asthma, but it’s usually under control with an inhaler. No heart disease in my family.
Are you on any medications for your asthma?
Yes, I use an inhaler when needed, and it’s been helping.
How’s your activity level? Do you exercise regularly?
I don’t exercise much at the moment because of work and personal commitments.
Does anyone in your family have a history of lung disease or heart problems?
My mother had asthma as well, but no one else in my family has heart problems.
"""

conversation_24 = """
Good morning! How are you feeling today?
I’ve been experiencing a lot of bloating and discomfort in my stomach after meals.
How long has this been happening?
It started about three weeks ago, and it happens almost every time I eat.
Have you had any nausea, vomiting, or changes in bowel movements?
No nausea or vomiting, but I do feel very gassy and bloated after eating.
Do you have a history of acid reflux, IBS, or food intolerances?
I’ve had occasional acid reflux in the past, but I’ve never been diagnosed with IBS or food intolerances.
Are you taking any medications right now?
I take an antacid when needed, but that’s about it.
How’s your diet and lifestyle? Do you eat a lot of processed or fatty foods?
I’ve been eating a lot of fast food lately due to my hectic schedule.
Does anyone in your family have a history of gastrointestinal issues?
My father has a history of acid reflux, and my grandmother had IBS.
"""

conversation_25 = """
Hello! What brings you in today?
I’ve been dealing with some leg cramps at night, and it’s affecting my sleep.
How long has this been happening?
It started about a month ago, and it happens almost every night now.
Have you had any other symptoms, such as swelling or redness in the legs?
No, just the cramps. They seem to come on suddenly and wake me up in the middle of the night.
Do you have a history of circulatory issues or varicose veins?
I’ve never had issues with circulation, but my grandmother had varicose veins.
Are you on any medications right now?
I take a multivitamin and calcium supplements for my bones.
How’s your activity level? Do you stand or sit for long periods?
I sit most of the day for my job, but I do try to get up and move around every hour.
Does anyone in your family have a history of leg cramps or circulatory issues?
My grandmother had varicose veins, and my mother has occasional leg cramps.
"""

conversation_26 = """
Hi there! What seems to be the problem today?
I’ve been dealing with frequent heartburn and acid reflux, especially after meals.
How long have you been experiencing these symptoms?
It started about a month ago, and it’s been getting worse. I get heartburn almost every day now.
Do you have any other symptoms like chest pain or difficulty swallowing?
No chest pain, but I do feel a burning sensation in my throat and chest after meals.
Have you had any history of GERD (gastroesophageal reflux disease) or ulcers?
I’ve had acid reflux in the past, but it’s never been this bad. No ulcers.
Are you currently taking any medications for this condition?
I take an over-the-counter antacid sometimes, but it doesn’t seem to help much anymore.
How’s your diet? Do you eat a lot of spicy or acidic foods?
I do tend to eat a lot of spicy foods, and I drink coffee and soda regularly.
Does anyone in your family have a history of acid reflux or gastrointestinal issues?
Yes, my father has GERD, and my mother has a history of ulcers.
"""

conversation_27 = """
Good morning! What brings you in today?
I’ve been feeling a lot of pain and tightness in my chest, especially after physical activity.
How long have you been experiencing this pain?
It started about a week ago, and it’s been happening more frequently now.
Have you experienced any shortness of breath, dizziness, or nausea with the chest pain?
Yes, I feel short of breath and lightheaded, especially when the pain occurs.
Do you have a history of heart disease or high blood pressure?
I have high blood pressure, but no history of heart disease.
Are you currently on any medications for your blood pressure?
Yes, I take medication to control my blood pressure, but it’s been fluctuating lately.
How would you describe your physical activity? Do you exercise regularly?
I don’t exercise regularly. I’ve been more sedentary lately due to work.
Does anyone in your family have a history of heart disease or high blood pressure?
Yes, my father has high blood pressure, and my uncle had a heart attack.
"""

conversation_28 = """
Hello! What brings you in today?
I’ve been feeling a lot of pressure and pain in my lower back and hips.
How long have you been experiencing this pain?
It started about a month ago, and it’s been progressively getting worse.
Have you noticed any changes in your mobility or any numbness in your legs?
I haven’t had any numbness, but I do feel stiff and have trouble bending down.
Do you have a history of back problems or hip issues?
I’ve had some minor back pain in the past, but nothing this severe.
Are you taking any pain medications or treatments for this condition?
I’ve been using over-the-counter pain relievers, but they don’t seem to help much.
How’s your daily activity level? Do you do any heavy lifting or physical activity?
I don’t lift heavy things, but I sit a lot at my desk job and don’t get much exercise.
Does anyone in your family have a history of joint or back problems?
Yes, my mother has arthritis in her hips and has had back problems in the past.
"""

conversation_29 = """
Hi, what brings you in today?
I’ve been experiencing frequent urination, especially at night, and a constant feeling of urgency.
How long has this been happening?
It started about two weeks ago, and it’s been happening more often recently.
Have you noticed any pain, blood, or a burning sensation when urinating?
No pain or blood, but I do feel some mild discomfort in my lower abdomen.
Do you have a history of urinary tract infections (UTIs) or bladder issues?
I’ve had UTIs in the past, but this feels different. I don’t have any infections at the moment.
Are you on any medications or taking anything for this condition?
No, I haven’t started any medication for this yet.
How would you describe your hydration and diet? Do you drink enough water?
I drink water, but I often drink a lot of coffee and sugary drinks, especially during work hours.
Does anyone in your family have a history of bladder or urinary issues?
Yes, my grandmother had issues with frequent urination and incontinence.
"""

conversation_30 = """
Hello, how are you feeling today?
I’ve been experiencing some pain and swelling in my ankles, especially after standing for long periods.
How long has this been happening?
It started about a few weeks ago, and it’s been progressively worse. My ankles get very swollen by the end of the day.
Have you had any redness, warmth, or difficulty moving your ankles?
No redness or warmth, but they do feel stiff and tight.
Do you have a history of circulatory problems, varicose veins, or blood clots?
I don’t have a history of blood clots, but I’ve had varicose veins in the past.
Are you currently taking any medications or treatments for your ankles?
I’ve been using over-the-counter anti-inflammatory creams, but they don’t seem to help much.
How’s your activity level? Do you sit or stand for long periods during the day?
I stand a lot at work, and I don’t take many breaks to sit down.
Does anyone in your family have a history of circulatory issues or swollen ankles?
Yes, my mother has had swollen ankles and varicose veins for years.
"""

conversation_31 = """
Hello! What can I help you with today?
I’ve been feeling unusually tired and fatigued lately, no matter how much rest I get.
How long have you been feeling this way?
It’s been happening for the past three weeks. I’m exhausted even after a full night’s sleep.
Have you noticed any changes in your appetite or weight?
I’ve lost some weight recently, and I’ve also had little appetite.
Do you have any history of thyroid disorders or other chronic illnesses?
I was diagnosed with hypothyroidism a few years ago, and I take medication for it.
Are you currently on any other medications?
Yes, I take thyroid medication every day, but it doesn’t seem to help much with the fatigue.
How would you describe your daily activity level and stress levels?
I don’t have a lot of physical activity, but I’m under a lot of stress at work.
Does anyone in your family have a history of thyroid or autoimmune disorders?
Yes, my mother has Hashimoto’s disease.
"""

conversation_32 = """
Good afternoon! What brings you in today?
I’ve been experiencing joint pain and stiffness, especially in my knees and wrists.
How long have you been having these symptoms?
It started about a month ago, and it’s been getting progressively worse.
Have you had any swelling, redness, or difficulty moving the affected joints?
Yes, my wrists feel swollen, and I find it hard to bend them.
Do you have a history of arthritis or other joint conditions?
I was diagnosed with rheumatoid arthritis about five years ago, but I haven’t had issues recently.
Are you currently taking any medications for arthritis?
Yes, I take a daily anti-inflammatory medication to manage my condition.
How would you describe your lifestyle? Do you stay physically active?
I try to stay active, but the pain has been limiting my exercise routine lately.
Does anyone in your family have a history of arthritis or autoimmune conditions?
Yes, my father has osteoarthritis.
"""

conversation_33 = """
Hi there! How are you feeling today?
I’ve been experiencing a persistent cough and some shortness of breath for the past few weeks.
How long have you been dealing with these symptoms?
It started about three weeks ago, and the cough just doesn’t seem to go away.
Do you have any chest pain, wheezing, or a fever?
I don’t have chest pain or a fever, but I do feel a tightness in my chest sometimes.
Have you ever been diagnosed with asthma, COPD, or any respiratory conditions?
I have mild asthma, but I haven’t had any flare-ups in years.
Are you currently taking any medications for asthma or allergies?
Yes, I use an inhaler occasionally when I feel short of breath.
How’s your daily activity level? Do you experience symptoms with physical exertion?
I do feel winded more easily these days, especially when walking upstairs.
Does anyone in your family have a history of asthma or lung diseases?
Yes, my father had COPD, and my sister has asthma.
"""

conversation_34 = """
Hello! What brings you in today?
I’ve been noticing some difficulty with my vision, especially when reading or looking at screens.
How long have you been having this issue?
It started a few weeks ago, and it seems to be getting worse. I have to squint a lot.
Do you have any headaches or eye pain?
I’ve been having some headaches, especially after using my phone or computer for a while.
Do you wear glasses or contact lenses?
I wear glasses for reading, but I’ve had them for years and never had this issue before.
Have you been to an eye doctor recently or had any eye exams?
It’s been about a year since my last exam, but I haven’t had any problems up until now.
How would you describe your daily screen time and activities?
I spend a lot of time in front of a computer screen and on my phone for work.
Does anyone in your family have a history of vision problems?
Yes, my mother has cataracts, and my grandmother had glaucoma.
"""

conversation_35 = """
Good morning! What brings you in today?
I’ve been having some pain and discomfort in my abdomen, especially after meals.
How long have you been experiencing this pain?
It started about two weeks ago, and the pain seems to come and go.
Have you had any nausea, vomiting, or changes in bowel movements?
I’ve had some nausea, but no vomiting. My bowel movements have been normal.
Do you have a history of ulcers, IBS, or other gastrointestinal issues?
I was diagnosed with IBS a few years ago, but I haven’t had any major issues since then.
Are you currently taking any medications for IBS or stomach problems?
I’m not on any medication right now, but I used to take fiber supplements.
How would you describe your diet? Do you eat a lot of greasy or spicy foods?
I tend to eat a lot of fast food and spicy foods, especially when I’m stressed.
Does anyone in your family have a history of gastrointestinal issues?
Yes, my father has a history of ulcers.
"""

conversation_36 = """
Hello, how are you feeling today?
I’ve been feeling lightheaded and dizzy, especially when I stand up quickly.
How long have you been experiencing these symptoms?
It started about a week ago, and it happens several times a day.
Have you experienced any fainting or blurred vision with the dizziness?
I haven’t fainted, but I do feel like I might pass out sometimes, and my vision gets blurry.
Do you have any history of low blood pressure, anemia, or heart conditions?
I have low blood pressure, and I’ve had anemia in the past, but it’s been under control.
Are you currently on any medications for these conditions?
I take iron supplements for anemia, and I try to manage my blood pressure through diet.
How would you describe your activity levels? Do you exercise regularly?
I don’t exercise much, as I feel too tired and dizzy.
Does anyone in your family have a history of blood pressure issues or anemia?
Yes, my mother has low blood pressure, and my grandmother had anemia.
"""

conversation_37 = """
Hi! What brings you in today?
I’ve been feeling a lot of joint pain in my fingers and toes, especially when I wake up in the morning.
How long have you been experiencing these symptoms?
It’s been about a month now, and it’s been worse in the mornings.
Have you noticed any swelling or redness in the joints?
Yes, my fingers feel swollen and stiff, and they’re hard to move in the mornings.
Do you have a history of arthritis or any autoimmune conditions?
I was diagnosed with rheumatoid arthritis about two years ago, but it’s been under control with medication.
Are you currently taking any medications for rheumatoid arthritis?
Yes, I take disease-modifying antirheumatic drugs (DMARDs) and pain relievers.
How would you describe your physical activity? Do you have any difficulty with mobility?
I try to stay active, but the pain makes it difficult to exercise.
Does anyone in your family have a history of autoimmune diseases?
Yes, my aunt has lupus, and my mother has osteoarthritis.
"""

conversation_38 = """
Good afternoon! What brings you in today?
I’ve been having trouble sleeping, waking up several times throughout the night.
How long has this been going on?
It started about two weeks ago, and it’s been affecting my daily energy levels.
Do you have any trouble falling asleep, or is it more about staying asleep?
I have trouble staying asleep. I wake up frequently and then struggle to fall back asleep.
Have you been under any increased stress or anxiety lately?
Yes, work has been very stressful, and I’ve been feeling anxious about deadlines.
Are you currently taking any medications for sleep or anxiety?
I haven’t taken anything for sleep or anxiety, though I’ve tried over-the-counter sleep aids with little success.
How would you describe your daily routine and activity levels?
I have a sedentary job and don’t get much exercise, so I’m not very active during the day.
Does anyone in your family have a history of sleep disorders or anxiety?
Yes, my father has insomnia, and my sister struggles with anxiety.
"""

conversation_39 = """
Hello! What brings you in today?
I’ve been dealing with frequent headaches, especially around my forehead and temples.
How long have you been having these headaches?
It started about a month ago, and it happens almost every day now.
Do you experience any nausea, light sensitivity, or visual disturbances with the headaches?
I get some nausea with the headaches, but no visual disturbances.
Have you had any history of migraines or tension headaches?
I’ve had migraines in the past, but these headaches feel different. They’re more constant.
Are you currently taking any medications for these headaches?
I’ve been using over-the-counter pain relievers, but they don’t seem to help much.
How’s your sleep pattern? Do you get enough rest?
I don’t sleep well. I often wake up during the night, and I don’t feel fully rested.
Does anyone in your family have a history of migraines or chronic headaches?
Yes, my mother has migraines, and my brother suffers from tension headaches.
"""

conversation_40 = """
Good morning! What brings you in today?
I’ve been having some pain and discomfort in my chest, especially after eating.
How long has this been going on?
It’s been happening for the past two weeks, and it seems to get worse after meals.
Have you noticed any shortness of breath, dizziness, or nausea with the chest pain?
No shortness of breath, but I do feel a bit nauseous sometimes after eating.
Do you have a history of heart disease or gastrointestinal issues?
I don’t have heart disease, but I do have a history of acid reflux and GERD.
Are you currently on any medications for heartburn or reflux?
Yes, I take a proton pump inhibitor, but it doesn’t seem to work well anymore.
How’s your diet? Do you eat a lot of spicy or fatty foods?
I eat a lot of spicy foods, and I’ve noticed that they make the pain worse.
Does anyone in your family have a history of acid reflux or heart disease?
Yes, my father has acid reflux, and my grandmother had heart disease.
"""

conversation_41 = """
Hi there! What can I help you with today?
I’ve been having a lot of lower back pain that doesn’t seem to go away, especially after sitting for long periods.
How long have you been experiencing this pain?
It started about a month ago, and it’s been getting worse lately.
Does the pain radiate down your legs, or is it just localized in your back?
It’s mostly localized in my lower back, but sometimes I feel a dull ache going down my right leg.
Do you have any history of back problems or injuries?
I’ve had a couple of minor back strains in the past, but nothing this persistent.
Are you currently on any medications or treatments for your back pain?
I’ve been taking over-the-counter pain relievers, but they don’t seem to help much.
How’s your daily activity level? Do you exercise regularly?
I don’t exercise much, and I sit at a desk most of the day for work.
Does anyone in your family have a history of back pain or sciatica?
My father has had chronic back pain for years, and my brother recently had sciatica.
"""

conversation_42 = """
Hello! What brings you in today?
I’ve been experiencing dizziness and occasional ringing in my ears.
How long have you been experiencing these symptoms?
It started about two weeks ago, and the dizziness comes and goes.
Do you have any history of ear infections, hearing loss, or balance issues?
I’ve had ear infections in the past, but no major hearing loss.
Have you noticed any changes in your hearing or difficulty balancing?
The ringing is constant, and I feel a little off balance when I stand up.
Are you currently on any medications or treatments?
I’m not on any medications right now, but I’ve been using ear drops occasionally.
How’s your overall health? Any other issues like headaches or vision problems?
I’ve had a few headaches lately, but nothing too severe.
Does anyone in your family have a history of ear or balance problems?
My mother has a history of vertigo and tinnitus.
"""

conversation_43 = """
Hi! How can I help you today?
I’ve been feeling more anxious and stressed than usual, especially in social situations.
How long have you been feeling this way?
It’s been going on for a few months now, and it’s been getting progressively worse.
Have you noticed any physical symptoms, like increased heart rate, sweating, or shaking?
Yes, my heart races and I sweat a lot during these episodes.
Do you have a history of anxiety or panic attacks?
I’ve had mild anxiety in the past, but I’ve never experienced panic attacks like this before.
Are you currently taking any medications for anxiety or depression?
No, I’m not on any medication, but I’ve been trying to manage it with relaxation techniques.
How would you describe your daily routine? Are you under a lot of stress?
I have a stressful job, and I don’t get much time to unwind or relax.
Does anyone in your family have a history of anxiety or other mental health issues?
Yes, my mother has generalized anxiety disorder.
"""

conversation_44 = """
Good afternoon! What can I assist you with today?
I’ve been having some pain in my throat and difficulty swallowing for the past week.
How long have you been dealing with this issue?
It started last week, and it’s been getting worse, especially when I try to swallow solids.
Do you have a sore throat, fever, or any other symptoms like coughing or runny nose?
I don’t have a fever or runny nose, but I do have a sore throat and a dry cough.
Have you had any issues with your tonsils or a history of frequent throat infections?
I’ve had tonsillitis before, but it’s been years since I had any problems.
Are you currently on any medications for this condition?
No, I’ve been trying to manage it with warm teas and throat lozenges.
How’s your overall health otherwise? Any other concerns?
Other than this, I feel fine, but the pain is making it hard to eat.
Does anyone in your family have a history of throat problems or infections?
Yes, my father had recurrent tonsillitis when he was younger.
"""

conversation_45 = """
Hello! What brings you in today?
I’ve been experiencing a lot of abdominal bloating and discomfort, especially after eating.
How long have you been feeling this way?
It’s been happening for the past few weeks, and it seems to get worse after meals.
Do you have any changes in bowel movements, diarrhea, or constipation?
I’ve had some constipation, but no diarrhea.
Do you have any history of irritable bowel syndrome (IBS) or gastrointestinal conditions?
I’ve had IBS in the past, but I haven’t had any major flare-ups in years.
Are you currently on any medications for IBS or digestive issues?
No, I stopped taking medication a while ago because I thought I had it under control.
How would you describe your diet? Do you eat a lot of dairy or high-fat foods?
I tend to eat a lot of processed foods, but I try to avoid dairy as it sometimes triggers bloating.
Does anyone in your family have a history of digestive disorders?
Yes, my mother has IBS, and my father has had several gastrointestinal issues.
"""

conversation_46 = """
Hi! What brings you in today?
I’ve been having a constant feeling of pressure in my chest and shortness of breath.
How long have you been feeling this way?
It started about a week ago, and it feels like there’s something heavy on my chest.
Do you have any pain in your chest or any discomfort radiating to your arms or neck?
No pain, just pressure and tightness. I don’t feel any discomfort in my arms or neck.
Have you had any heart issues in the past, like high blood pressure or chest pain?
I have high blood pressure, but it’s usually well-controlled with medication.
Are you currently taking any medications for heart issues or other conditions?
Yes, I’m on medication for high blood pressure, but I haven’t had any issues until now.
How would you describe your physical activity? Do you exercise regularly?
I don’t exercise regularly, but I try to stay active by walking.
Does anyone in your family have a history of heart disease or high blood pressure?
Yes, both of my parents have high blood pressure.
"""

conversation_47 = """
Hello! What brings you in today?
I’ve been having some issues with my skin lately. I’ve noticed a lot of redness and dryness on my face.
How long has this been going on?
It started about two weeks ago, and it hasn’t improved despite using moisturizer.
Have you tried any new skin products or treatments recently?
I started using a new moisturizer, but I’m not sure if that’s causing the issue.
Do you have any history of skin conditions like eczema or rosacea?
I have mild eczema, but it’s never been this bad before.
Are you currently using any medications or treatments for your skin?
I’m using the new moisturizer and a hydrocortisone cream for the eczema.
How’s your general health? Do you feel well otherwise?
I’ve been feeling fine otherwise, but the redness and dryness are bothering me.
Does anyone in your family have a history of skin conditions?
Yes, my mother has rosacea, and my father had eczema.
"""

conversation_48 = """
Good morning! How can I assist you today?
I’ve been having difficulty breathing and a constant tightness in my chest.
How long have you been experiencing these symptoms?
It’s been happening for the past few days, and it feels worse when I’m lying down.
Do you have any history of asthma, COPD, or heart disease?
I have asthma, but it’s usually well-controlled with my inhaler.
Have you been using your inhaler more frequently lately?
Yes, I’ve had to use it more often in the past few days.
Are you experiencing any other symptoms like coughing, wheezing, or swelling?
I’ve had some coughing, but no swelling or wheezing.
How’s your daily activity level? Do you feel winded during physical exertion?
I do feel winded more easily, and I get short of breath faster than usual.
Does anyone in your family have a history of respiratory or heart problems?
Yes, my father had COPD, and my grandmother had heart disease.
"""

conversation_49 = """
Hello! What brings you in today?
I’ve been experiencing severe headaches and blurry vision, especially when I try to read or focus on close objects.
How long have you had these symptoms?
It’s been a few weeks now, and the headaches seem to worsen in the evening.
Do you have a history of migraines, vision problems, or eye strain?
I’ve had occasional migraines, but nothing this intense.
Have you tried any treatments or medications for the headaches?
I’ve taken some over-the-counter painkillers, but they don’t seem to help.
How’s your daily routine? Do you spend a lot of time on screens or reading?
I work on a computer all day, and I read a lot for my job.
Does anyone in your family have a history of eye conditions or migraines?
Yes, my mother has glaucoma, and my sister suffers from migraines.
"""

conversation_50 = """
Hi! What brings you in today?
I’ve been having a persistent cough and a sore throat for the last two weeks.
How long have you had these symptoms?
It’s been going on for about two weeks, and it’s not improving.
Do you have a fever, congestion, or runny nose?
I don’t have a fever, but I’ve had a little congestion.
Have you had any issues with your tonsils or throat in the past?
I had strep throat when I was younger, but I haven’t had any major throat issues since then.
Are you currently taking any medications for this condition?
I’ve been using throat lozenges, but they haven’t helped much.
How’s your daily activity level? Do you feel weak or fatigued?
I feel pretty tired, and I’m not as energetic as usual.
Does anyone in your family have a history of throat infections or respiratory issues?
Yes, my father has had multiple episodes of strep throat.
"""



questions = [
    "Summarize the patient's current and past health issues.",
    "List the patient's chief complaints with comma separated value.",
    "Summarize the patient's current symptoms and their medical background, including past diagnoses.",
    "Summarize the patient's past medical history.",
    "Summarize the patient's hospitalization and surgical history.",
    "Extract and summarize the patient's gynecological history from the context.",
    "Summarize the patient's lifestyle and social activity.",
    "Summarize the patient's family history.",
    "Summarize the patient's have any known allergies, hypersensitivities, or adverse reactions to medications?",
    "Summarize the patient's previous investigations of patient."
]


answers_conversation_1 = [
    "The patient has been feeling fatigued with muscle aches for over a month, waking up exhausted despite 8 hours of sleep. They also report unintentional weight loss, decreased appetite, and occasional shortness of breath while climbing stairs.",
    "Fatigue, muscle aches, unintentional weight loss, decreased appetite, shortness of breath.",
    "The patient has persistent fatigue, muscle aches, and weight loss. These symptoms are similar to what they experienced last year but have persisted for over a month this time. No known past diagnoses are mentioned.",
    "Past diagnosis: Clinical, Appendicitis, Cured (Appendix removed 5 years ago).",
    "Appendicitis, Appendix removal, 5 years ago.",
    "No data available in the context.",
    "Sedentary lifestyle, desk job, irregular meals, lacks exercise, insufficient vegetable intake.",
    "Father has diabetes, mother had anemia in her youth.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned. Basic tests are planned to rule out other causes of headaches."
]

answers_conversation_2 = [
    "The patient is experiencing irregular menstrual cycles for the past six months, with missed months and prolonged bleeding. They also report fatigue, weight gain (especially around the midsection), and mood changes.",
    "Irregular menstrual cycles, fatigue, weight gain, mood changes (anxiety).",
    "The patient reports irregular menstrual cycles, fatigue, weight gain, and mood changes. They had irregular cycles in the past but stabilized in their twenties. No past diagnoses are mentioned.",
    "No data available in the context.",
    "Severe flu, hospitalization, 3 years ago.",
    "Irregular menstrual cycles, past history of irregularity in youth, stabilized in mid-twenties.",
    "Sedentary lifestyle, works from home, inconsistent diet, no exercise.",
    "Mother had fibroids, sister diagnosed with polycystic ovary syndrome (PCOS).",
    "No known allergies or reactions to medications.",
    "No recent investigations mentioned."
]

answers_conversation_3 = [
    "The patient has been feeling exhausted for the past two months, despite adequate sleep. They have a history of hypothyroidism, for which they are taking medication. The patient's appetite has decreased, and they’ve lost some weight.",
    "Tiredness, decreased appetite, weight loss.",
    "The patient is experiencing persistent fatigue and weight loss. They have hypothyroidism, diagnosed a few years ago, and are currently on medication (levothyroxine). The symptoms of fatigue have lasted for two months.",
    "Clinical, Hypothyroidism, Undertreatment.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Sedentary lifestyle, desk job, high stress levels, no exercise.",
    "Mother, Thyroid problems, No specific age mentioned, Father, High blood pressure, No specific age mentioned.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_4 = [
    "The patient is experiencing persistent acne around the chin and jawline, which started about a month ago and has been worsening. They had acne as a teenager, but it cleared up in their twenties. There are no other significant skin issues mentioned.",
    "Acne around chin and jawline, dry skin.",
    "The patient has developed persistent acne over the past month, which is getting worse. They have a history of acne in their teenage years, which cleared up in their twenties. No other major health issues are mentioned.",
    "Past diagnosis: Clinical, Acne, Cured (Acne as a teenager, cleared up in twenties).",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Stressful job, frequently touches face while working.",
    "No family history of acne or related conditions mentioned.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_5 = [
    "The patient is experiencing frequent tightness in the chest and shortness of breath, along with lightheadedness and swelling in the ankles. They have a history of asthma, which has been controlled for years. No recent chest pain or significant issues are mentioned.",
    "Tightness in chest, shortness of breath, lightheadedness, swelling in ankles.",
    "The patient has tightness in the chest and shortness of breath, occurring frequently in the past week. They have a history of asthma, which has been controlled for years, and are using an inhaler as needed. No chest pain or dizziness is reported.",
    "Past diagnosis: Clinical, Asthma, Controlled.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Used to smoke but quit five years ago, no current exercise routine.",
    "Father, Heart attack in forties, No specific age mentioned, Grandmother, COPD, No specific age mentioned.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_6 = [
    "The patient has noticed a firm lump in their neck, which has been there for two weeks. It’s not painful unless pressed, and it hasn’t grown in size. The patient had a mild sore throat last month, but it cleared up. They have a history of a normal thyroid check.",
    "Lump in neck, mild sore throat (last month).",
    "The patient has a lump in the neck, which has been present for two weeks. It’s firm to the touch but not painful unless pressed. No issues with thyroid or lymph nodes were found previously.",
    "Past diagnosis: Clinical, Thyroid check, Normal.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "No details on lifestyle or social activity provided.",
    "Aunt, Thyroid cancer, Remission for several years.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_7 = [
    "The patient has been experiencing lower back pain for the past month, which worsens after prolonged sitting. There’s no numbness or tingling in the legs, but they do feel stiffness in the lower back when standing up. They have no major injuries but have previously lifted heavy objects, which may have strained their back.",
    "Lower back pain, stiffness after sitting, no numbness or tingling.",
    "The patient has lower back pain, which started a month ago and has been gradually increasing. They experience stiffness in the lower back when standing after sitting, but no numbness or tingling in the legs. No major injuries are mentioned, but lifting heavy objects may have strained the back.",
    "No past diagnoses mentioned.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Sedentary lifestyle, desk job, minimal exercise.",
    "Mother, Back issues, Grandfather, Osteoarthritis.",
    "Occasionally takes ibuprofen for back pain.",
    "No previous investigations mentioned."
]

answers_conversation_8 = [
    "The patient has been experiencing frequent urination, especially at night, for the past two weeks. They wake up three to four times a night to use the bathroom, but there’s no pain or burning sensation. They don’t have a history of UTIs or kidney problems, and they are not currently on any medications.",
    "Frequent urination, especially at night, no pain or burning sensation.",
    "The patient has frequent urination at night, which has been happening for two weeks. They wake up three to four times each night. No pain or burning sensation while urinating. No history of UTIs or kidney problems is mentioned.",
    "No past diagnoses mentioned.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Consumes a lot of coffee during the day, drinks more alcohol recently, no mention of physical activity.",
    "Mother has diabetes, no urinary problems in family.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_9 = [
    "The patient has been experiencing joint pain in their knees and elbows, making it difficult to move around. The pain started a month ago and has been gradually getting worse, especially after sitting for long periods. There’s some swelling in the knees but no redness or warmth.",
    "Joint pain in knees and elbows, swelling in knees.",
    "The patient has joint pain in the knees and elbows that started a month ago. The pain is worse after sitting for long periods. There’s swelling in the knees, but no redness or warmth is noted. The patient has a family history of rheumatoid arthritis and osteoarthritis.",
    "No past diagnoses mentioned.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Sedentary lifestyle, sits at work, no physical activity mentioned.",
    "Mother has rheumatoid arthritis, grandfather had osteoarthritis.",
    "Occasionally takes ibuprofen for joint pain.",
    "No previous investigations mentioned."
]

answers_conversation_10 = [
    "The patient has been experiencing chest discomfort, sometimes feeling like pressure in the chest, mostly after eating or when stressed. They also feel lightheaded when the discomfort occurs. They have a history of heartburn and occasionally use an antacid when it gets bad.",
    "Chest discomfort, lightheadedness, heartburn.",
    "The patient experiences chest discomfort and lightheadedness after eating or during stress. There’s a history of heartburn, which the patient treats with over-the-counter antacids. No pain radiating to the arms or shortness of breath is reported.",
    "Past diagnosis: Clinical, Heartburn, Occasional treatment with antacids.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Consumes a lot of spicy food, skips meals during busy days, drinks a lot of coffee.",
    "Father had heart disease, had a heart attack five years ago, currently doing fine.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_11 = [
    "The patient has sharp abdominal pain on the right side, which started three days ago and has been worsening. The pain is uncomfortable when moving, and they have experienced mild fever and bloating. No past gastrointestinal issues or appendicitis reported.",
    "Sharp abdominal pain on the right side, mild fever, bloating.",
    "The patient reports sharp abdominal pain, especially on the right side, which started three days ago and has worsened. They have a mild fever and some bloating, but no history of gastrointestinal issues or appendicitis is mentioned.",
    "No past gastrointestinal issues or appendicitis history.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "The patient eats more fast food due to a busy schedule.",
    "No family history of gastrointestinal issues.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]


answers_conversation_12 = [
    "The patient has been experiencing severe headaches for the past week, which are worse in the evenings. They also feel lightheaded and sensitive to bright lights. They have a history of stress-induced headaches, but these are more intense and frequent.",
    "Severe headaches, lightheadedness, sensitivity to light.",
    "The patient experiences severe headaches starting a week ago, which happen daily, especially in the evenings. They also feel lightheaded and are sensitive to bright lights. They report a history of stress-related headaches but mention that these feel different and more intense.",
    "No history of migraines or neurological problems.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "The patient has poor sleep and high stress levels due to work.",
    "No family history of migraines or neurological issues.",
    "Occasionally takes ibuprofen for the headaches.",
    "No previous investigations mentioned."
]


answers_conversation_13 = [
    "The patient has been experiencing lower back pain for the past few weeks, which has been worsening. They don’t experience numbness, tingling, or weakness, but the pain is worse after sitting or bending. There’s a history of mild back issues, but no major injuries.",
    "Lower back pain, no numbness or tingling.",
    "The patient reports worsening lower back pain that started a few weeks ago. The pain is exacerbated by sitting or bending, and there is no numbness or tingling. There is a history of mild back issues but no major injuries.",
    "No major back issues or injuries in the past.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Sedentary lifestyle, desk job, little physical activity.",
    "Mother has chronic back pain.",
    "Occasionally takes acetaminophen for the back pain.",
    "No previous investigations mentioned."
]


answers_conversation_14 = [
    "The patient has been feeling anxious for the past month, with symptoms affecting their concentration and sleep. They experience physical symptoms like a racing heart and sweating, especially under stress. The anxiety is more intense than past episodes.",
    "Anxiety, racing heart, sweating, trouble concentrating and sleeping.",
    "The patient has been feeling increasingly anxious for the past month, affecting their ability to concentrate and sleep. They also experience physical symptoms like a racing heart and sweating, particularly under stress. The anxiety is more intense than previous episodes.",
    "No past history of anxiety or depression mentioned, but increased intensity now.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Busy lifestyle with work and family responsibilities, little exercise or downtime.",
    "Father has anxiety and depression, sister has generalized anxiety disorder (GAD).",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]


answers_conversation_15 = [
    "The patient has been experiencing irregular periods for the past two months, with changes in frequency and heaviness. They also report weight gain, increased emotional sensitivity, and oily skin. No history of polycystic ovary syndrome (PCOS), but their mother had fibroids and early menopause.",
    "Irregular periods, weight gain, emotional sensitivity, oily skin.",
    "The patient reports irregular periods, with changes in frequency and heaviness, over the past two months. Additionally, they’ve gained weight, become more emotional, and their skin has become oily. No history of PCOS, but their mother had fibroids and early menopause.",
    "No history of PCOS, but mother had fibroids and early menopause.",
    "No hospitalization or surgeries mentioned.",
    "No data available in the context.",
    "Sedentary lifestyle, poor diet with fast food.",
    "No family history of PCOS, mother had fibroids and early menopause.",
    "No known allergies or reactions to medications.",
    "No previous investigations mentioned."
]

answers_conversation_16 = [
    "The patient has been experiencing constant fatigue for the past two weeks, despite adequate rest. There is a history of anemia but no thyroid problems at the moment.",
    "Fatigue, lethargy.",
    "The patient reports feeling constantly drained with no improvement after rest. Past medical history includes anemia, but current iron levels are normal, and there are no thyroid issues.",
    "Past diagnosis: Clinical, Anemia, Cured.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, Stressful job, No exercise.",
    "Sister, Chronic Fatigue Syndrome, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_17 = [
    "The patient has been experiencing frequent nosebleeds, especially in the mornings. There is a history of mild allergies but no bleeding disorders or major sinus issues.",
    "Frequent nosebleeds.",
    "The patient reports frequent nosebleeds, especially in the morning, with no additional symptoms. The patient has mild allergies but no other notable health issues.",
    "Past diagnosis: Clinical, Mild allergies, Cured.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, No exercise.",
    "No family history of nosebleeds.",
    "Allergy, Antihistamines, Mild reaction, Active, Low severity.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_18 = [
    "The patient has been experiencing right shoulder and neck pain for the past few days, worsened by lifting heavy boxes. No history of musculoskeletal problems.",
    "Shoulder pain, Neck pain, Limited movement.",
    "The patient reports pain and stiffness in the right shoulder and neck, aggravated by lifting heavy boxes. No prior musculoskeletal issues.",
    "Past diagnosis: Clinical, No history of neck or shoulder problems, Cured.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, Desk job, No exercise.",
    "No family history of musculoskeletal problems.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_19 = [
    "The patient experiences dizziness and lightheadedness when standing up quickly. The patient has a history of low blood pressure, which is currently well-controlled.",
    "Dizziness, Lightheadedness, Low blood pressure.",
    "The patient reports dizziness and lightheadedness, especially upon standing up. The patient has a history of low blood pressure, which is controlled.",
    "Past diagnosis: Clinical, Low blood pressure, Controlled.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, Forgetting to drink water, Stressful job.",
    "Father, Low blood pressure, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_20 = [
    "The patient experiences knee and hip pain, particularly after walking or standing. The patient has no known diagnosis of arthritis, though a family history of osteoarthritis exists.",
    "Knee pain, Hip pain, Swelling, Redness.",
    "The patient has knee and hip pain with swelling and redness. There is a family history of osteoarthritis but no personal diagnosis.",
    "Past diagnosis: Suspected, Osteoarthritis, N/A.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, No exercise, Desk job.",
    "Mother, Osteoarthritis, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_21 = [
    "The patient has been experiencing hair loss for the past month, which is causing significant concern. There is a history of thyroid problems and hormonal imbalances.",
    "Hair loss, Stress, Thyroid issues.",
    "The patient has had significant hair shedding every day for the past month. There is a history of thyroid issues, and the patient is concerned it may be hormonal.",
    "Past diagnosis: Clinical, Thyroid problems, Under control.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, High stress levels, Poor diet.",
    "Mother, Thyroid issues, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_22 = [
    "The patient has been experiencing pain and a burning sensation while urinating, along with mild lower abdominal discomfort. There is a history of urinary tract infections (UTIs), but the current symptoms feel different.",
    "Painful urination, Burning sensation, Lower abdominal discomfort.",
    "The patient reports burning sensation while urinating and mild discomfort in the lower abdomen. They have had UTIs in the past, but the current symptoms feel different.",
    "Past diagnosis: Clinical, UTI, Cured.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, No exercise.",
    "No family history of urinary issues.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_23 = [
    "The patient has been feeling short of breath, particularly when engaging in physical activity. They have asthma but no history of heart disease.",
    "Shortness of breath, Lightheadedness, Asthma.",
    "The patient reports feeling short of breath, particularly during physical activity, with occasional lightheadedness. The patient has asthma but no history of heart disease.",
    "Past diagnosis: Clinical, Asthma, Controlled.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, No exercise.",
    "Mother, Asthma, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_24 = [
    "The patient has been experiencing bloating and discomfort in the stomach after meals, with no nausea or vomiting. The patient has a history of acid reflux, and their diet has consisted of a lot of fast food recently.",
    "Bloating, Stomach discomfort, Acid reflux.",
    "The patient reports bloating and discomfort in the stomach after meals, with no nausea or vomiting. The patient has a history of acid reflux and has been eating a lot of fast food.",
    "Past diagnosis: Clinical, Acid reflux, Controlled.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, Fast food diet, No exercise.",
    "Father, Acid reflux, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_25 = [
    "The patient has been dealing with leg cramps at night, waking them up frequently. There is no history of circulatory issues, but the patient’s grandmother had varicose veins.",
    "Leg cramps, Night pain, No swelling.",
    "The patient reports frequent leg cramps at night, waking them up. There is no history of circulatory problems, though the patient’s grandmother had varicose veins.",
    "Past diagnosis: Clinical, No circulatory issues, Cured.",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Inactive, Sedentary job, No exercise.",
    "Grandmother, Varicose veins, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_26 = [
    "The patient has been experiencing frequent heartburn and acid reflux, which started a month ago and has been getting worse. They have a history of acid reflux but no ulcers.",
    "Heartburn, Acid reflux, Burning sensation in throat and chest.",
    "The patient reports persistent heartburn and acid reflux after meals, with a burning sensation in the throat and chest. They have a history of acid reflux but no ulcers.",
    "Past diagnosis: GERD (Provisional, No ulcers).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Spicy foods and coffee regularly.",
    "Father, GERD, N/A, Mother, History of ulcers, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_27 = [
    "The patient has been experiencing chest pain and tightness, especially after physical activity. They also report shortness of breath and lightheadedness. The patient has high blood pressure.",
    "Chest pain, Tightness in chest, Shortness of breath, Lightheadedness.",
    "The patient reports chest pain, tightness after physical activity, and shortness of breath. They have high blood pressure but no history of heart disease.",
    "Past diagnosis: High blood pressure (Clinical, Final).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Lack of regular exercise due to work.",
    "Father, High blood pressure, N/A, Uncle, Heart attack, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_28 = [
    "The patient has been experiencing pain and pressure in the lower back and hips, which started a month ago and has been getting worse. They have a history of minor back pain.",
    "Lower back pain, Hip pain, Stiffness, Difficulty bending down.",
    "The patient has progressive pain and stiffness in the lower back and hips. They have had minor back pain in the past but nothing as severe as this.",
    "Past diagnosis: None (History of minor back pain).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Desk job, No exercise.",
    "Mother, Arthritis in hips, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_29 = [
    "The patient has been experiencing frequent urination, especially at night, and a constant feeling of urgency. They have a history of UTIs but no current infection.",
    "Frequent urination, Urgency, Mild discomfort in lower abdomen.",
    "The patient reports frequent urination and urgency, particularly at night, with mild discomfort in the lower abdomen. They have a history of UTIs but no current infections.",
    "Past diagnosis: UTI (Clinical, Final, History of UTIs).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Moderate, Drinks coffee and sugary drinks during work.",
    "Grandmother, Frequent urination, Incontinence, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_30 = [
    "The patient has been experiencing pain and swelling in the ankles, especially after standing for long periods. They have a history of varicose veins but no circulatory issues currently.",
    "Pain in ankles, Swelling in ankles, Stiffness in ankles.",
    "The patient reports pain and swelling in the ankles, especially after standing for long periods. They have a history of varicose veins but no blood clots.",
    "Past diagnosis: Varicose veins (Clinical, Final).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, High, Stands a lot at work with few breaks.",
    "Mother, Swollen ankles and varicose veins, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_31 = [
    "The patient reports fatigue and tiredness, which started three weeks ago. They also mention weight loss and decreased appetite, with a history of hypothyroidism.",
    "Fatigue, Weight loss, Low appetite.",
    "The patient has been feeling fatigued for the past three weeks, with weight loss and little appetite. They have a history of hypothyroidism.",
    "Past diagnosis: Hypothyroidism (Clinical, Current).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Under stress at work.",
    "Mother, Hashimoto’s disease, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_32 = [
    "The patient has been experiencing joint pain and stiffness in the knees and wrists, starting a month ago and worsening since. They have a history of rheumatoid arthritis.",
    "Joint pain, Stiffness in knees and wrists, Swollen wrists.",
    "The patient reports joint pain, stiffness, and swelling in the wrists, with a history of rheumatoid arthritis.",
    "Past diagnosis: Rheumatoid arthritis (Clinical, Current).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Limited, Difficulty with exercise due to pain.",
    "Father, Osteoarthritis, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_33 = [
    "The patient has been experiencing a persistent cough and shortness of breath for the past three weeks. They have a history of mild asthma.",
    "Cough, Shortness of breath, Tightness in chest.",
    "The patient reports a persistent cough and shortness of breath, with occasional chest tightness. They have a history of mild asthma.",
    "Past diagnosis: Asthma (Mild, Controlled).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Reduced, Gets winded easily with exertion.",
    "Father, COPD, N/A, Sister, Asthma, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_34 = [
    "The patient reports difficulty with vision, especially when reading or looking at screens, starting a few weeks ago. They have a history of glasses use for reading.",
    "Difficulty reading, Squinting, Headaches after screen use.",
    "The patient has trouble reading and squints frequently, with headaches after screen use. They have glasses for reading but have not experienced this issue before.",
    "Past diagnosis: None (History of using glasses for reading).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Long screen time for work.",
    "Mother, Cataracts, N/A, Grandmother, Glaucoma, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_35 = [
    "The patient reports abdominal pain and discomfort after meals for the past two weeks. They have a history of IBS.",
    "Abdominal pain, Discomfort after meals, No vomiting.",
    "The patient has pain and discomfort after meals, with a history of IBS but no significant issues recently.",
    "Past diagnosis: IBS (Clinical, History).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Eats fast food and spicy foods.",
    "Father, Ulcers, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_36 = [
    "The patient reports lightheadedness and dizziness when standing up quickly, starting a week ago. They have a history of low blood pressure and anemia.",
    "Lightheadedness, Dizziness, Blurry vision.",
    "The patient has lightheadedness, dizziness, and blurry vision upon standing. They have low blood pressure and a history of anemia.",
    "Past diagnosis: Low blood pressure, Anemia (Clinical, Controlled).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Low, Does not exercise regularly.",
    "Mother, Low blood pressure, N/A, Grandmother, Anemia, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_37 = [
    "The patient has joint pain in their fingers and toes, especially in the mornings. They have a history of rheumatoid arthritis.",
    "Joint pain, Swollen fingers and toes, Stiffness in the morning.",
    "The patient reports joint pain and stiffness in their fingers and toes, especially in the morning. They have a history of rheumatoid arthritis.",
    "Past diagnosis: Rheumatoid arthritis (Clinical, Current).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Limited, Difficulty exercising due to pain.",
    "Aunt, Lupus, N/A, Mother, Osteoarthritis, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_38 = [
    "The patient has been experiencing trouble sleeping, with frequent awakenings during the night, which has been affecting their energy levels. They are under stress at work.",
    "Trouble sleeping, Frequent awakenings, Low energy.",
    "The patient has difficulty staying asleep, with frequent awakenings at night and low energy. They report work-related stress.",
    "Past diagnosis: None (No sleep disorders diagnosed).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Work-related stress.",
    "Father, Insomnia, N/A, Sister, Anxiety, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_39 = [
    "The patient reports frequent headaches around their forehead and temples, starting a month ago. They have a history of migraines.",
    "Headaches, Nausea, Light sensitivity.",
    "The patient has frequent headaches with nausea and light sensitivity. They have a history of migraines but these headaches feel constant.",
    "Past diagnosis: Migraines (Clinical, History).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Difficulty sleeping.",
    "Mother, Migraines, N/A, Brother, Tension headaches, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_40 = [
    "The patient reports chest pain after meals, which started two weeks ago. They have a history of acid reflux and GERD.",
    "Chest pain, Nausea after eating, No shortness of breath.",
    "The patient experiences chest pain and nausea after meals, with a history of acid reflux and GERD.",
    "Past diagnosis: GERD (Clinical, Current).",
    "No hospitalizations or surgeries reported.",
    "No data available in the context.",
    "Physical activity, Sedentary, Eats spicy foods.",
    "Father, Acid reflux, N/A, Grandmother, Heart disease, N/A.",
    "No allergies or hypersensitivities reported.",
    "The patient has not mentioned any recent investigations."
]

answers_conversation_41 = [
    "The patient is experiencing persistent lower back pain that has been getting worse over the past month. Pain sometimes radiates down to the right leg. The patient has a history of minor back strains but no major back issues.",
    "Lower back pain, radiating pain in the right leg, pain worsening with prolonged sitting.",
    "The patient has persistent lower back pain that occasionally radiates to the right leg. No significant past back injuries except for minor strains. Pain is not relieved by over-the-counter pain relievers.",
    "Back strains (Past Clinical), no significant back problems.",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Physical activity: Sedentary job, minimal exercise, long sitting hours.",
    "Father: Chronic back pain, Brother: Sciatica.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_42 = [
    "The patient is experiencing dizziness and tinnitus, with a history of ear infections but no major hearing loss.",
    "Dizziness, ringing in ears, off balance, occasional headaches.",
    "The patient has constant ringing in the ears, dizziness, and balance issues. No recent hearing loss but a history of ear infections.",
    "Ear infections (Past Clinical), no hearing loss.",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Not specified.",
    "Mother: History of vertigo, Tinnitus.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_43 = [
    "The patient is feeling more anxious and stressed, especially in social situations, with physical symptoms like heart racing and sweating. Mild anxiety in the past but no panic attacks.",
    "Increased anxiety, heart racing, sweating, stress in social situations.",
    "The patient experiences increased anxiety, heart racing, and sweating in social situations. No panic attacks but has had mild anxiety in the past.",
    "Mild anxiety (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Stressful job, limited relaxation time.",
    "Mother: Generalized anxiety disorder.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_44 = [
    "The patient has throat pain and difficulty swallowing, especially with solids, and a history of tonsillitis.",
    "Throat pain, difficulty swallowing, dry cough, sore throat.",
    "The patient has a sore throat and dry cough, with difficulty swallowing solids. Has had tonsillitis in the past, but is not currently taking any medications.",
    "Tonsillitis (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Not specified.",
    "Father: Recurrent tonsillitis.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_45 = [
    "The patient reports abdominal bloating and discomfort, especially after meals, with a history of IBS.",
    "Abdominal bloating, discomfort after meals, constipation.",
    "The patient has abdominal bloating, discomfort after meals, and constipation. They have a history of IBS but are not currently on medication for it.",
    "Irritable bowel syndrome (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Diet: Processed foods, avoids dairy.",
    "Mother: IBS, Father: Gastrointestinal issues.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_46 = [
    "The patient is experiencing chest pressure and shortness of breath, with a history of well-controlled high blood pressure.",
    "Chest pressure, shortness of breath, tightness in chest.",
    "The patient is experiencing chest pressure and shortness of breath. They have a history of high blood pressure, which is usually well-controlled.",
    "High blood pressure (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Physical activity: Walking, but not regular exercise.",
    "Both parents: High blood pressure.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_47 = [
    "The patient has redness and dryness on their face, which started two weeks ago and has not improved despite using moisturizer. The patient has a history of mild eczema.",
    "Redness, dryness on face, mild eczema.",
    "The patient has redness and dryness on the face, which has not improved. They have a history of mild eczema, currently using moisturizer and hydrocortisone cream.",
    "Eczema (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Not specified.",
    "Mother: Rosacea, Father: Eczema.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_48 = [
    "The patient reports difficulty breathing and tightness in the chest, with a history of asthma, which is usually well-controlled.",
    "Difficulty breathing, tightness in chest, frequent inhaler use.",
    "The patient is experiencing difficulty breathing, chest tightness, and increased use of their inhaler. They have a history of asthma.",
    "Asthma (Past Clinical), Inhaler use more frequent recently.",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Physical activity: Shortness of breath with exertion.",
    "Father: COPD, Grandmother: Heart disease.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_49 = [
    "The patient is experiencing severe headaches and blurry vision, especially when reading or focusing on close objects, with a history of occasional migraines.",
    "Severe headaches, blurry vision, difficulty focusing, migraines.",
    "The patient has severe headaches and blurry vision, particularly when reading or focusing on close objects. They have a history of occasional migraines.",
    "Migraines (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Work on computer all day, frequent reading.",
    "Mother: Glaucoma, Sister: Migraines.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]

answers_conversation_50 = [
    "The patient has a persistent cough and sore throat for the last two weeks, with a history of strep throat in childhood.",
    "Persistent cough, sore throat, mild congestion.",
    "The patient has a persistent cough and sore throat, with mild congestion. They had strep throat when younger, but no major throat issues since.",
    "Strep throat (Past Clinical).",
    "No relevant surgical or hospitalization history.",
    "Not applicable (male patient).",
    "Not specified.",
    "Father: History of strep throat.",
    "No allergies or hypersensitivities mentioned.",
    "No previous investigations mentioned."
]


data = []


for i in range(len(questions)):
    data.append([questions[i], conversation_1.strip(), answers_conversation_1[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_2.strip(), answers_conversation_2[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_3.strip(), answers_conversation_3[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_4.strip(), answers_conversation_4[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_5.strip(), answers_conversation_5[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_6.strip(), answers_conversation_6[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_7.strip(), answers_conversation_7[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_8.strip(), answers_conversation_8[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_9.strip(), answers_conversation_9[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_10.strip(), answers_conversation_10[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_11.strip(), answers_conversation_11[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_12.strip(), answers_conversation_12[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_13.strip(), answers_conversation_13[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_14.strip(), answers_conversation_14[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_15.strip(), answers_conversation_15[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_16.strip(), answers_conversation_16[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_17.strip(), answers_conversation_17[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_18.strip(), answers_conversation_18[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_19.strip(), answers_conversation_19[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_20.strip(), answers_conversation_20[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_21.strip(), answers_conversation_21[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_22.strip(), answers_conversation_22[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_23.strip(), answers_conversation_23[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_24.strip(), answers_conversation_24[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_25.strip(), answers_conversation_25[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_26.strip(), answers_conversation_26[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_27.strip(), answers_conversation_27[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_28.strip(), answers_conversation_28[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_29.strip(), answers_conversation_29[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_30.strip(), answers_conversation_30[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_31.strip(), answers_conversation_31[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_32.strip(), answers_conversation_32[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_33.strip(), answers_conversation_33[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_34.strip(), answers_conversation_34[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_35.strip(), answers_conversation_35[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_36.strip(), answers_conversation_36[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_37.strip(), answers_conversation_37[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_38.strip(), answers_conversation_38[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_39.strip(), answers_conversation_39[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_40.strip(), answers_conversation_40[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_41.strip(), answers_conversation_41[i]])

for i in range(len(questions)):
    data.append([questions[i], conversation_42.strip(), answers_conversation_42[i]])

for i in range(len(questions)):
    data.append([questions[i], conversation_43.strip(), answers_conversation_43[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_44.strip(), answers_conversation_44[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_45.strip(), answers_conversation_45[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_46.strip(), answers_conversation_46[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_47.strip(), answers_conversation_47[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_48.strip(), answers_conversation_48[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_49.strip(), answers_conversation_49[i]])


for i in range(len(questions)):
    data.append([questions[i], conversation_50.strip(), answers_conversation_50[i]])


random.shuffle(data)


df = pd.DataFrame(data, columns=["Question", "Context", "Answer"])


df.to_csv("dataset_finetune_s.csv", index=False, encoding='utf-8')

print("Dataset generated and saved to 'doctor_patient_dataset.csv'.")
