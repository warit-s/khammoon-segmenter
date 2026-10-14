<objective>
Segment each `raw_text` into words using `|` as the delimiter by strictly following the rules below. Each segmented unit must represent a monomorphemic unit (คำมูล).
</objective>

<critical_constraints>
1. ATOMIC INTEGRITY: `''.join(tokenized.split('|'))` MUST exactly equal the `raw_text`.
2. Do not add, remove, change, or normalize any characters, including whitespace (spaces become `| |`).
3. Never produce tokens that are orthographically invalid in Thai (e.g., a token starting with `า` or ending with `เ`) unless the input was already invalid.
4. The only permitted modification is inserting `|`.
</critical_constraints>

<guidelines>
Apply these steps sequentially for every input.

**Step 1: Structural-level segmentation**
Split the text by syntactic/functional chunks. Preserve whitespace exactly.

Input: แม่ฝากฉันซื้อปลา แต่ตลาดปิด
Output: แม่|ฝาก|ฉัน|ซื้อ|ปลา| |แต่|ตลาด|ปิด

Input: ลลิษา มโนบาลเป็นไอดอลสาวชาวไทย
Output: ลลิษา| |มโนบาล|เป็น|ไอดอล|สาว|ชาว|ไทย

Input: ว้าย นั่นใคร
Output: ว้าย| |นั่น|ใคร

**Step 2: Non-Thai / punctuation / symbol handling**
* **KEEP (Do not split):**
    * Technical data (URLs, file paths, emails, tags, filenames)
        Input: ไปที่ www.google.com เลย
        Output: ไป|ที่| |www.google.com| |เลย
        Input: ชื่อไฟล์ screenshot_2.png ค่ะ
        Output: ชื่อ|ไฟล์| |screenshot_2.png| |ค่ะ
        Input: ส่งมาที่ mike@abc.co.uk ครับ
        Output: ส่ง|มา|ที่| |mike@abc.co.uk| |ครับ
    * Numeric values with commas or decimal points
        Input: เงิน 2,552.24 พันล้านบาท
        Output: เงิน| |2,552.24| |พัน|ล้าน|บาท
        Input: จำนวนยอด 18,603 บาท
        Output: จำนวน|ยอด| |18,603| |บาท
        Input: โรงเรียนอยู่ห่างออกไป 720.5 เมตร
        Output: โรง|เรียน|อยู่|ห่าง|ออก|ไป| |720.5| |เมตร
    * Section/heading numbers
        Input: 1. นกนางนวล
        Output: 1.| |นก|นาง|นวล
        Input: 3) นางสมร
        Output: 3)| |นาง|สมร
        Input: บทความที่ (๒) ระบุว่า
        Output: บท|ความ|ที่| |(๒)| |ระบุ|ว่า
    * Chemical compounds
	    Input: Ca(OH)2 + CO2
	    Output: Ca(OH)2| |+| |CO2

* **SPLIT (Separate into individual tokens):**
    * Foreign-language text (split on whitespace)
        Input: into the borderland
        Output: into| |the| |borderland
        Input: the 4th generation
        Output: the| |4th| |generation
        Input: People's Alliance for Democracy
        Output: People's| |Alliance| |for| |Democracy
        Input: That’s bad
        Output: That’s| |bad
    * Punctuation (parentheses, quotes)
        Input: บดินทรเดชา (สิงห์ สิงหเสนี)
        Output: บดินทรเดชา| |(|สิงห์| |สิงหเสนี|)
        Input: เขาร้องว่า “ช่วยด้วย”
        Output: เขา|ร้อง|ว่า| |“|ช่วย|ด้วย|”
    * Symbols (%, #, etc.) separated from adjacent text
        Input: มีขยะเพิ่มเป็น 25.50%
        Output: มี|ขยะ|เพิ่ม|เป็น| |25.50|%
        Input: #วันหยุด #ทะเล
        Output: #|วัน|หยุด| |#|ทะเล
        Input: ไม่มีอีกแล้ว.... เงินเก็บ
        Output: ไม่|มี|อีก|แล้ว|.|.|.|.| |เงิน|เก็บ
        Input: บริษัท Yahoo! มีกำไร $300,000-400,000
        Output: บริษัท| |Yahoo|!| |มี|กำไร| |$|300,000|-|400,000
        Input: ลดราคา 25%
        Output: ลด|ราคา| |25|%
    * Mathematical/chemical symbols (+, -, =, ~, ^, --->, etc.) separated from adjacent text
        Input: Ca(OH)2 + CO2 ---> CaCO3 + H2O + heat
        Output: Ca(OH)2| |+| |CO2| |--->| |CaCO3| |+| |H2O| |+| |heat
        Input: [(18*67)+15]-17^2 = ~42.22
        Output: [|(|18|*|67|)|+|15|]|-|17|^|2| |=| |~|42.22


**Step 3: Thai-character handling**
* **3.1 Assume correct orthography:** Segment based on intended valid spelling.
    Input: ศรีษะมาร
    Output: ศรีษะ|มาร
    Input: พะยักพะเยิด
    Output: พะยัก|พะเยิด
    Input: กะหรอมกะแหรม
    Output: กะ|หรอม|กะ|แหรม
    Input: กระเพราผัดไก่
    Output: กระเพรา|ผัด|ไก่
* **3.2 Abbreviations & Punctuation:**
    * Single entity rank/unit/place -> KEEP. Stacked distinct credentials -> SPLIT.
        Input: ผบ.ทบ.สั่งการว่า
        Output: ผบ.ทบ.|สั่ง|การ|ว่า
        Input: คณะอนุกก.
        Output: คณะ|อนุกก.
        Input: ผบ.ภ.อ.คง กล่าวว่า
        Output: ผบ.ภ.|อ.|คง| |กล่าว|ว่า
        Input: รศ.ดร.พญ.วีณา
        Output: รศ.|ดร.|พญ.|วีณา
        Input: พ.ศ. 2569
        Output: พ.ศ.| |2569
    * Ellipsis (ฯ, ฯลฯ) -> SPLIT (Exception: ฯพณฯ is KEEP).
        Input: กรุงเทพฯ
        Output: กรุง|เทพ|ฯ
        Input: กล้วย ส้ม ฯลฯ
        Output: กล้วย| |ส้ม| |ฯลฯ
        Input: รอต้อนรับ ฯพณฯ นายกรัฐมนตรี
        Output: รอ|ต้อน|รับ| |ฯพณฯ| |นายก|รัฐมนตรี
    * ไม้ยมก (ๆ) -> SPLIT.
        Input: รักมากๆๆ
        Output: รัก|มาก|ๆ|ๆ
* **3.3 Context Resolution:** Read the full sentence to determine meaning BEFORE Step 4. Tests in Step 4 MUST use this contextually determined meaning.
    Input: ถ้าจะให้ตัดเสื้อก็วัดรอบอกมา
    Output: ถ้า|จะ|ให้|ตัด|เสื้อ|ก็|วัด|รอบ|อก|มา
    Input: คงไม่มีใครจะมารอบอกเธอหรอก
    Output: คง|ไม่|มี|ใคร|จะ|มา|รอ|บอก|เธอ|หรอก
    Input: พายุพัดเข้าฝั่งทำลายบ้านเรือน
    Output: พายุ|พัด|เข้า|ฝั่ง|ทำลาย|บ้าน|เรือน
    Input: ตัดกระดาษแล้วทำลายเป็นรูปดาว
    Output: ตัด|กระดาษ|แล้ว|ทำ|ลาย|เป็น|รูป|ดาว

**Step 4: Decide monomorphemic units (คำมูล) vs. compound words**
Evaluate each remaining chunk. Stop at the first category that applies.

* **Category 1: Mixed origins -> SPLIT**
    Input: เคมีภัณฑ์
    Output: เคมี|ภัณฑ์
    Input: พระเก้าอี้
    Output: พระ|เก้าอี้
    Input: ราชวัง
    Output: ราช|วัง
    Input: ผลไม้
    Output: ผล|ไม้
    Input: ซุปกิมจิ
    Output: ซุป|กิมจิ
    Input: คุณค่าของราชวังเก่า
    Output: คุณ|ค่า|ของ|ราช|วัง|เก่า
    Input: อุตสาหกรรมเคมีภัณฑ์
    Output: อุตสาหกรรม|เคมี|ภัณฑ์
    Input: เวชระเบียน
    Output: เวช|ระเบียน
    Input: พลเมือง
    Output: พล|เมือง

* **Category 2: All Pali-Sanskrit**
    * True สมาส-สนธิ (Meaning flows back->front): KEEP.
        Input: ประวัติศาสตร์
        Output: ประวัติศาสตร์
        Input: แพทยสภา
        Output: แพทยสภา
        Input: พุทธศาสนา
        Output: พุทธศาสนา
        Input: มหาบัณฑิต
        Output: มหาบัณฑิต
    * Thai-syntax order (front->back) or คำซ้อน: SPLIT.
        Input: ศาสนาพุทธ
        Output: ศาสนา|พุทธ
        Input: ฤทธิ์ณรงค์
        Output: ฤทธิ์|ณรงค์
        Input: ประชาชน
        Output: ประชา|ชน

* **Category 3: Loanwords / transliterations**
    * Interpreted back->front: KEEP.
        Input: วิปครีม
        Output: วิปครีม
        Input: คอนกรีตบล็อก
        Output: คอนกรีตบล็อก
    * Interpreted front->back: SPLIT.
        Input: คุกกี้ช็อกโกแลต
        Output: คุกกี้|ช็อกโกแลต
        Input: เค้กวานิลลา
        Output: เค้ก|วานิลลา

* **Category 4: Native Thai words**
    * **4-A Sound Reduction (กร่อนเสียง): KEEP.**
        Input: ตะวัน
        Output: ตะวัน
        Input: มะม่วง
        Output: มะม่วง
        Input: มะพร้าว
        Output: มะพร้าว
        Input: กินมะม่วงกะมะพร้าว
        Output: กิน|มะม่วง|กะ|มะพร้าว
    * **4-B Meaningless affix syllables: SPLIT (unless it is a fused prefix like ระ-, กระ-, ประ-).**
        Input: ตาเตอ
        Output: ตา|เตอ
        Input: ร้อนเริ้น
        Output: ร้อน|เริ้น
        Input: กระป๋งกระเป๋า
        Output: กระป๋ง|กระเป๋า
        Input: พะรุงพะรัง
        Output: พะ|รุง|พะ|รัง
        Input: กะหนุงกะหนิง
        Output: กะ|หนุง|กะ|หนิง
        Input: ดี๊ดี
        Output: ดี๊|ดี
        Input: อากาศร้อนเริ้น
        Output: อากาศ|ร้อน|เริ้น
        Input: ของวางพะรุงพะรัง
        Output: ของ|วาง|พะ|รุง|พะ|รัง
        Input: กะหนุงกะหนิงกันทั้งวัน
        Output: กะ|หนุง|กะ|หนิง|กัน|ทั้ง|วัน
        *(Fused Prefixes to KEEP):*
        Input: ระรื่น
        Output: ระรื่น
        Input: กระโดด
        Output: กระโดด
        Input: ประเดี๋ยว
        Output: ประเดี๋ยว
        Input: กระผม
        Output: กระผม
        Input: ประเดี๋ยวกระผมนำมาให้
        Output: ประเดี๋ยว|กระผม|นำ|มา|ให้
    * **4-C Onomatopoeia (คำเลียนเสียงธรรมชาติ)**
        Base ABAB -> KEEP. Explicit doubling/interleaved -> SPLIT.
        Input: คิกคัก
        Output: คิกคัก
        Input: ฉึกฉัก
        Output: ฉึกฉัก
        Input: ยุกยิก
        Output: ยุกยิก
        Input: ตึกตัก
        Output: ตึกตัก
        Input: พึบพับ
        Output: พึบพับ
        Input: คิกคิกคักคัก
        Output: คิก|คิก|คัก|คัก
        Input: พึบพึบพับพับ
        Output: พึบ|พึบ|พับ|พับ
        Input: ฉึกฉึกฉักฉัก
        Output: ฉึก|ฉึก|ฉัก|ฉัก
        Input: ยุกยุกยิกยิก
        Output: ยุก|ยุก|ยิก|ยิก
        Input: ตึกตึกตักตัก
        Output: ตึก|ตึก|ตัก|ตัก
        Input: กระฉึกกระฉาก
        Output: กระ|ฉึก|กระ|ฉาก
    * **4-D Meaning substitution / insertion test (using Step 3.3 context)**
        If meaningful word can be inserted or substituted -> SPLIT. Else -> KEEP.
        Input: รถไฟ
        Output: รถ|ไฟ
        Input: ทิศตะวันออก
        Output: ทิศ|ตะวัน|ออก
        Input: มือถือ
        Output: มือ|ถือ
        Input: ทำลาย (meaning: to make patterns)
        Output: ทำ|ลาย
        Input: ทำลาย (meaning: to destroy)
        Output: ทำลาย
        Input: นาฬิกา
        Output: นาฬิกา
        Input: กระตือรือร้น
        Output: กระตือรือร้น

* **Category 5: Unknown origin or persistent uncertainty -> Pronunciation test**
    If pronunciation unchanged -> SPLIT. If pronunciation changes -> KEEP.
    Input: ขะมักเขม้น
    Output: ขะมัก|เขม้น
    Input: จิ้มลิ้ม
    Output: จิ้ม|ลิ้ม
    Input: จุกจิก
    Output: จุก|จิก
    Input: พระสงฆ์
    Output: พระ|สงฆ์
    Input: ภาคภูมิ
    Output: ภาค|ภูมิ
    Input: อนึ่ง
    Output: อนึ่ง
</guidelines>

<few_shot_examples>
Input: โรงเรียนสอนคนตาบอด
Output: โรง|เรียน|สอน|คน|ตา|บอด

Input: พลเมืองดีศรีสังคม
Output: พล|เมือง|ดี|ศรี|สังคม

Input: รัฐบาลประกาศนโยบาย
Output: รัฐบาล|ประกาศ|นโยบาย

Input: นายสมชายทำงานที่ปารีส
Output: นาย|สม|ชาย|ทำ|งาน|ที่|ปารีส

Input: กลุ่มองค์กรต่างๆ ในสังคมนานาชาติ
Output: กลุ่ม|องค์กร|ต่าง|ๆ| |ใน|สังคม|นานาชาติ

Input: มหาวิทยาลัยเชียงใหม่
Output: มหาวิทยาลัย|เชียง|ใหม่

Input: คุณค่าของชุมชนราชวังเก่า
Output: คุณ|ค่า|ของ|ชุม|ชน|ราช|วัง|เก่า

Input: อุตสาหกรรมไฟฟ้าในประเทศไทย
Output: อุตสาหกรรม|ไฟ|ฟ้า|ใน|ประเทศ|ไทย

Input: ผกก.ภ.อ.คง กล่าว
Output: ผกก.ภ.|อ.|คง| |กล่าว

Input: พล.ต.อ.จงรัก รอง ผบ.ตร.(ปป.3)
Output: พล.ต.อ.|จง|รัก| |รอง| |ผบ.ตร.|(|ปป.|3|)

Input: ผบ.สน.ทองหล่อบุกจับกุมผศ.ดร.วัฒนแสงประเสริฐ
Output: ผบ.|สน.|ทอง|หล่อ|บุก|จับ|กุม|ผศ.|ดร.|วัฒน|แสง|ประเสริฐ

Input: ผบ.สส.พงส์เพ็ชรปฏิเสสเหตุการณ์ดังกล่าว
Output: ผบ.สส.|พงส์|เพ็ชร|ปฏิเสส|เหตุ|การณ์|ดัง|กล่าว

Input: กป.อพช.ภาคอีสาน 12-14/12/2559
Output: กป.อพช.|ภาค|อีสาน| |12|-|14|/|12|/|2559

Input: วิธีการอันชาญฉลาด
Output: วิธี|การ|อัน|ชาญ|ฉลาด

Input: การแปรรูปผลิตผลจากพืช
Output: การ|แปร|รูป|ผลิตผล|จาก|พืช

Input: โรงงานผลิตผลไม้ฟรีซดราย
Output: โรง|งาน|ผลิต|ผล|ไม้|ฟรีซดราย

Input: การสัมมนาเชิงวิชาการ
Output: การ|สัมมนา|เชิง|วิชาการ

Input: วิชาการท่องเที่ยวและการโรงแรม
Output: วิชา|การ|ท่อง|เที่ยว|และ|การ|โรง|แรม

Input: ประเดี๋ยวกระผมนำมาให้ขอรับ
Output: ประเดี๋ยว|กระผม|นำ|มา|ให้|ขอ|รับ

Input: เรียนเชิญคุณสมบัติ เมทะนี
Output: เรียน|เชิญ|คุณ|สมบัติ| |เมทะนี

Input: ๑) ผู้สมัครต้องมีคุณสมบัติดังนี้
Output: ๑)| |ผู้|สมัคร|ต้อง|มี|คุณสมบัติ|ดัง|นี้

Input: อัตราเร็ว ๑.๒-๑.๕ เมตร/วินาที
Output: อัตรา|เร็ว| |๑.๒|-|๑.๕| |เมตร|/|วินาที

Input: “เอ... ใช่ไหมนะ? ใช่สิ!” ดิฉันสับสน
Output: “|เอ|.|.|.| |ใช่|ไหม|นะ|?| |ใช่|สิ|!|”| |ดิฉัน|สับ|สน

Input: เขา [อดีตนายกฯ] ขออภัยโทษ
Output: เขา| |[|อดีต|นายก|ฯ|]| |ขอ|อภัยโทษ

Input: บ้านขนาด > 70 เมตร^2 แต่ < 150 เมตร^2 ราคา THB500,000 + ภาษี 7%
Output: บ้าน|ขนาด| |>| |70| |เมตร|^|2| |แต่| |<| |150| |เมตร|^|2| |ราคา| |THB|500,000| |+| |ภาษี| |7|%

Input: บริษัท Yahoo! มีมูลค่า $300,000-400,000
Output: บริษัท| |Yahoo|!| |มี|มูล|ค่า| |$|300,000|-|400,000

Input: ฝากซื้อแป้งสาลี, ไข่ไก่, เนย, น้ำตาล ฯลฯ
Output: ฝาก|ซื้อ|แป้ง|สาลี|,| |ไข่|ไก่|,| |เนย|,| |น้ำ|ตาล| |ฯลฯ

Input: ติด #ผัดไทเส้นจันทน์ แล้วแอดไลน์ @lineID
Output: ติด| |#|ผัด|ไท|เส้น|จันทน์| |แล้ว|แอด|ไลน์| |@lineID

Input: [(18*67)+15]-17^2 = ~42.22
Output: [|(|18|*|67|)|+|15|]|-|17|^|2| |=| |~|42.22

Input: ประชาชนต่อสู้เพื่อประชาธิปไตย
Output: ประชา|ชน|ต่อ|สู้|เพื่อ|ประชาธิปไตย

Input: พิธีพระราชทานปริญญาบัตร จุฬาลงกรณ์มหาวิทยาลัย
Output: พิธี|พระ|ราชทาน|ปริญญาบัตร| |จุฬาลงกรณ์มหาวิทยาลัย

Input: เด็กกำพร้าในบ้านโกโรโกโส
Output: เด็ก|กำพร้า|ใน|บ้าน|โกโรโกโส

Input: กำพร้าไว้แน่น ๆ อย่าให้หล่น
Output: กำ|พร้า|ไว้|แน่น| |ๆ| |อย่า|ให้|หล่น

Input: สุภาพสตรีหัวเราะคิกคัก
Output: สุภาพสตรี|หัว|เราะ|คิกคัก

Input: รำคาญเสียงคิกคิกคักคัก
Output: รำคาญ|เสียง|คิก|คิก|คัก|คัก

Input: 2542. สิทธิชุมชนและอัตลักษณ์ทางวัฒนธรรม: ชาติพันธุ์วรรณาในปทุมธานี [เอกสารสัมภาษณ์ประกอบเวทีเสวนา]. ชุมชนในสายธารโลกาภิวัตน์.
Output: 2542|.| |สิทธิ|ชุม|ชน|และ|อัตลักษณ์|ทาง|วัฒนธรรม|:| |ชาติพันธุ์วรรณา|ใน|ปทุมธานี| |[|เอกสาร|สัมภาษณ์|ประกอบ|เวที|เสวนา|]|.| |ชุม|ชน|ใน|สาย|ธาร|โลกาภิวัตน์|.

Input: นกโป่งวิดจะจับคู่ทำรังในช่วงฤดูฝน
Output: นก|โป่งวิด|จะ|จับ|คู่|ทำ|รัง|ใน|ช่วง|ฤดู|ฝน

Input: นกกาเหว่ามักจะแอบวางไข่ในรังของอีกา
Output: นก|กาเหว่า|มัก|จะ|แอบ|วาง|ไข่|ใน|รัง|ของ|อี|กา

Input: ถิ่นอาศัยของนกเค้าแมวมีหลากหลายคล้ายกับนกฮูก
Output: ถิ่น|อาศัย|ของ|นก|เค้า|แมว|มี|หลาก|หลาย|คล้าย|กับ|นก|ฮูก

Input: พนักงานขนส่งสินค้าขับรถรับ-ส่งพัสดุ
Output: พนัก|งาน|ขน|ส่ง|สิน|ค้า|ขับ|รถ|รับ|-|ส่ง|พัสดุ

Input: กระทรวงมหาดไทยมีหน้าที่บริหารราชการแผ่นดินส่วนกลาง
Output: กระทรวง|มหาด|ไทย|มี|หน้า|ที่|บริหาร|ราชการ|แผ่น|ดิน|ส่วน|กลาง

Input: นายจุติเสฏฐ์ไม่มีเบาะแสของทองคำแท่ง
Output: นาย|จุติ|เสฏฐ์|ไม่|มี|เบาะ|แส|ของ|ทอง|คำ|แท่ง

Input: บ้านขนาด > 70 เมตร^2 ราคา THB500,000 + ภาษี 7%
Output: บ้าน|ขนาด| |>| |70| |เมตร|^|2| |ราคา| |THB|500,000| |+| |ภาษี| |7|%

Input: ถนนเลียบคลองสอง
Output: ถนน|เลียบ|คลอง|สอง

Input: นายภาคภูมิเดินทางมา
Output: นาย|ภาค|ภูมิ|เดิน|ทาง|มา

Input: ทำงานขะมักเขม้น
Output: ทำ|งาน|ขะมัก|เขม้น

Input: อนึ่ง ขอแจ้งว่า
Output: อนึ่ง| |ขอ|แจ้ง|ว่า
</few_shot_examples>

<output_format>
For each JSON line provided, you must first output a `<reasoning>` block where you briefly identify ambiguous words, determine their etymology, and apply the Step 4 tests based on context. Then, output the exact JSON string with the `tokenized` key added.

Example response:
<reasoning>
- "เด็กกำพร้าในบ้านโกโรโกโส": "กำพร้า" acts as an adjective here (orphan), so it's kept intact per context. "โกโรโกโส" is a single fused concept. "เด็ก" and "ใน" and "บ้าน" split normally.
</reasoning>
{"raw_text": "เด็กกำพร้าในบ้านโกโรโกโส", "tokenized": "เด็ก|กำพร้า|ใน|บ้าน|โกโรโกโส"}
</output_format>

Process the following input:
<input>
[INSERT JSONL HERE]
</input>
