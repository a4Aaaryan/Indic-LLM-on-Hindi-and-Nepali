# Phase 3 Full Dataset Validation Report

## [PASS] No errors found.

### HINDI / TRAIN — 9600 examples
  - Tasks: {'direct': 5789, 'transitive': 2677, 'attribute': 1134}
  - Attributes: {'height': 1747, 'generic_order': 2285, 'age': 1768, 'quantity': 1944, 'price': 1856}
  - Question Categories: {'height': 1747, 'generic_order': 2285, 'age': 1768, 'possession_count': 900, 'substance_amount': 687, 'price': 1856, 'object_count': 357}
  - Chain Lengths: {2: 8074, 1: 1134, 3: 392}
  - Top Templates: [('H44', 505), ('H41', 474), ('H43', 451), ('H42', 428), ('H45', 427)]
  - Top Entities: [('साइकिल', 899), ('कंप्यूटर', 899), ('कलम', 851), ('गाड़ी', 849), ('घड़ी', 847)]
  - Tokens: min=21, max=45, mean=30.3, median=30.0

### HINDI / VAL — 1200 examples
  - Tasks: {'direct': 687, 'transitive': 371, 'attribute': 142}
  - Attributes: {'price': 216, 'generic_order': 313, 'height': 208, 'age': 217, 'quantity': 246}
  - Question Categories: {'price': 216, 'generic_order': 313, 'height': 208, 'age': 217, 'possession_count': 129, 'object_count': 37, 'substance_amount': 80}
  - Chain Lengths: {2: 1000, 1: 142, 3: 58}
  - Top Templates: [('H45', 68), ('H44', 66), ('H42', 62), ('H41', 60), ('H23', 58)]
  - Top Entities: [('राजू', 373), ('विजय', 367), ('अनीता', 362), ('सीता', 360), ('संजय', 344)]
  - Tokens: min=21, max=46, mean=32.0, median=32.0

### HINDI / TEST — 1200 examples
  - Tasks: {'direct': 625, 'transitive': 499, 'attribute': 76}
  - Attributes: {'quantity': 125, 'generic_order': 320, 'height': 247, 'price': 282, 'age': 226}
  - Question Categories: {'substance_amount': 53, 'generic_order': 320, 'height': 247, 'price': 282, 'age': 226, 'possession_count': 54, 'object_count': 18}
  - Chain Lengths: {2: 899, 3: 225, 1: 76}
  - Top Templates: [('H31', 163), ('H40', 156), ('H22', 125), ('H9', 58), ('H7', 46)]
  - Top Entities: [('दिनेश', 367), ('गिलास', 356), ('मनोज', 351), ('दीपक', 351), ('कैमरा', 350)]
  - Tokens: min=20, max=41, mean=28.1, median=28.0
  - Test Familiar/Held-out: 621 / 579 (target ~50/50)
  - Overall Unique Rendered Texts in hindi: 10961
  - Overall Unique Reasoning Instances in hindi: 10452

### NEPALI / TRAIN — 9600 examples
  - Tasks: {'direct': 5759, 'transitive': 2746, 'attribute': 1095}
  - Attributes: {'age': 1762, 'generic_order': 2356, 'height': 1754, 'price': 1890, 'quantity': 1838}
  - Question Categories: {'age': 1762, 'generic_order': 2356, 'height': 1754, 'price': 1890, 'substance_amount': 669, 'possession_count': 835, 'object_count': 334}
  - Chain Lengths: {2: 8115, 1: 1095, 3: 390}
  - Top Templates: [('N44', 507), ('N42', 483), ('N41', 479), ('N43', 448), ('N45', 439)]
  - Top Entities: [('टेबुल', 869), ('कम्प्युटर', 868), ('गाडी', 864), ('साइकल', 859), ('कुर्सी', 844)]
  - Tokens: min=22, max=41, mean=30.8, median=31.0

### NEPALI / VAL — 1200 examples
  - Tasks: {'direct': 703, 'transitive': 356, 'attribute': 141}
  - Attributes: {'price': 238, 'quantity': 225, 'age': 223, 'generic_order': 302, 'height': 212}
  - Question Categories: {'price': 238, 'substance_amount': 95, 'age': 223, 'object_count': 41, 'generic_order': 302, 'height': 212, 'possession_count': 89}
  - Chain Lengths: {2: 1005, 1: 141, 3: 54}
  - Top Templates: [('N41', 69), ('N43', 64), ('N44', 61), ('N24', 57), ('N42', 57)]
  - Top Entities: [('अस्मिता', 368), ('नवीन', 351), ('सृजना', 348), ('दिपेश', 346), ('टोपी', 326)]
  - Tokens: min=23, max=40, mean=32.2, median=32.0

### NEPALI / TEST — 1200 examples
  - Tasks: {'direct': 628, 'transitive': 501, 'attribute': 71}
  - Attributes: {'height': 233, 'generic_order': 300, 'price': 291, 'age': 267, 'quantity': 109}
  - Question Categories: {'height': 233, 'generic_order': 300, 'price': 291, 'age': 267, 'possession_count': 61, 'substance_amount': 33, 'object_count': 15}
  - Chain Lengths: {2: 872, 3: 257, 1: 71}
  - Top Templates: [('N31', 173), ('N40', 153), ('N22', 149), ('N7', 56), ('N9', 52)]
  - Top Entities: [('विनोद', 375), ('विकास', 365), ('संगीता', 361), ('पुष्पा', 359), ('सुरज', 358)]
  - Tokens: min=21, max=40, mean=31.0, median=31.0
  - Test Familiar/Held-out: 599 / 601 (target ~50/50)
  - Overall Unique Rendered Texts in nepali: 10937
  - Overall Unique Reasoning Instances in nepali: 10432
