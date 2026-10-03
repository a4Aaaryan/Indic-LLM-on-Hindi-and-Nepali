# Phase 3 Template Bank

This bank contains strictly audited templates for Phase 3 dataset generation.

## Hindi Templates

### Height / Comparison
*Compatible Entities: People, Buildings, Trees, Objects (with height)*
*Task Type: Direct Comparison*
* H1: `{E1} की लंबाई {E2} से ज़्यादा है।` (E1 > E2)
* H2: `{E2} का कद {E1} से कम है।` (E2 < E1) [Entities: People only]
* H3: `{E2} और {E3} दोनों, {E1} से लंबे हैं।` (E2 > E1, E3 > E1)
* H4: `{E1}, {E2} के मुकाबले कद में छोटा है।` (E1 < E2) [Entities: People only]
* H5: `{E1} की लंबाई {E2} के मुकाबले कम है।` (E1 < E2)
* H6: `{E1}, {E2} से ज़्यादा लंबा है।` (E1 > E2)
* H7: `{E1}, {E2} से लंबा है, पर {E3} दोनों से छोटा है।` (E1 > E2, E3 < E1, E3 < E2)
* H9: `{E1} की लंबाई {E2} से ज़्यादा है और {E3} से कम है।` (E1 > E2, E1 < E3)
* H10: `{E1} की तुलना में {E2} अधिक लंबा है।` (E2 > E1)
* H11: `{E1} का कद {E2} से ऊंचा है।` (E1 > E2) [Entities: People only]
* H12: `{E1} और {E2} में से {E1} लंबा है।` (E1 > E2)
* H14: `{E1} की ऊंचाई {E2} और {E3} दोनों से अधिक है।` (E1 > E2, E1 > E3)

### Age
*Compatible Entities: People, Animals, Historical Artifacts*
*Task Type: Attribute or Direct Comparison*
* H16: `{E1} की उम्र {V1} साल है।` (Attribute: Age = V1)
* H17: `{E1}, {E2} से उम्र में बड़ा है।` (E1 > E2)
* H18: `{E1} की उम्र {E2} से कम है।` (E1 < E2)
* H19: `{E1}, {E2} से उम्र में छोटा है, लेकिन {E3} उससे भी छोटा है।` (E1 < E2, E3 < E1)
* H20: `{E1} और {E2} की उम्र बराबर है।` (E1 = E2)
* H22: `{E1} की उम्र {E2} से अधिक लेकिन {E3} से कम है।` (E1 > E2, E1 < E3)
* H23: `{E1} और {E2} की उम्र समान है, जबकि {E3} दोनों से बड़ा है।` (E1 = E2, E3 > E1, E3 > E2)

### Price
*Compatible Entities: Products, Books, Vehicles, Properties*
*Task Type: Attribute or Direct Comparison*
* H24: `{E1} की कीमत {V1} रुपये है।` (Attribute: Price = V1)
* H25: `{E1}, {E2} से महंगा है।` (E1 > E2)
* H26: `{E1} की कीमत {E2} से कम है।` (E1 < E2)
* H27: `{E1} और {E2} में से {E1} अधिक महंगा है।` (E1 > E2)
* H29: `{E1} की कीमत {E2} और {E3} दोनों से अधिक है।` (E1 > E2, E1 > E3)
* H30: `{E1} की कीमत {E2} से अधिक लेकिन {E3} से कम है।` (E1 > E2, E1 < E3)
* H31: `{E1} और {E2} की कीमत समान है, जबकि {E3} की कीमत दोनों से अधिक है।` (E1 = E2, E3 > E1, E3 > E2)

### Quantity / Counts
*Compatible Entities: Collections of items, countable objects*
*Task Type: Attribute or Direct Comparison*
* H32: `{E1} के पास {V1} किताबें हैं।` (Attribute: Count = V1)
* H33: `{E1} के पास {E2} से अधिक किताबें हैं।` (E1 > E2)
* H34: `{E1} के पास {E2} से कम सामान है।` (E1 < E2)
* H35: `{E1} में {E2} की तुलना में अधिक मात्रा है।` (E1 > E2) [Entities: Uncountable substances]
* H36: `{E1} की संख्या {E2} से अधिक है।` (E1 > E2)
* H37: `{E1} और {E2} की संख्या बराबर है।` (E1 = E2)
* H38: `{E1} की मात्रा {E2} से अधिक लेकिन {E3} से कम है।` (E1 > E2, E1 < E3)

### Transitive / Ordering
*Compatible Entities: Abstract Concepts, Ranked Items, People*
*Task Type: Transitive Reasoning*
* H40: `{E1}, {E2} से बड़ा है और {E2}, {E3} से बड़ा है।` (E1 > E2, E2 > E3)
* H41: `{E1}, {E2} से छोटा है, जबकि {E3}, {E2} से बड़ा है।` (E1 < E2, E3 > E2)
* H42: `{E1}, {E2} से आगे है और {E3}, {E1} से आगे है।` (E1 > E2, E3 > E1)
* H43: `{E1}, {E2} से ऊपर है और {E3}, {E2} से नीचे है।` (E1 > E2, E3 < E2)
* H44: `{E1} और {E2} बराबर हैं, लेकिन {E3} दोनों से अधिक है।` (E1 = E2, E3 > E1)
* H45: `{E1}, {E2} से अधिक है और {E3}, {E1} से कम है।` (E1 > E2, E3 < E1)

---

## Nepali Templates

### Height / Comparison
*Compatible Entities: People, Buildings, Trees, Objects (with height)*
*Task Type: Direct Comparison*
* N1: `{E1} को लम्बाइ {E2} को भन्दा बढी छ।` (E1 > E2)
* N2: `{E2} को कद {E1} को भन्दा कम छ।` (E2 < E1) [Entities: People only]
* N3: `{E2} र {E3} दुवै {E1} भन्दा अग्ला छन्।` (E2 > E1, E3 > E1)
* N4: `{E1}, {E2} को तुलनामा होचो छ।` (E1 < E2) [Entities: People only]
* N5: `{E1} को लम्बाइ {E2} को तुलनामा कम छ।` (E1 < E2)
* N6: `{E1}, {E2} भन्दा अग्लो छ।` (E1 > E2)
* N7: `{E1}, {E2} भन्दा अग्लो छ, तर {E3} दुवैभन्दा होचो छ।` (E1 > E2, E3 < E1, E3 < E2)
* N9: `{E1} को लम्बाइ {E2} को भन्दा बढी छ र {E3} को भन्दा कम छ।` (E1 > E2, E1 < E3)
* N10: `{E1} को तुलनामा {E2} अग्लो छ।` (E2 > E1)
* N11: `{E1} को कद {E2} को भन्दा अग्लो छ।` (E1 > E2) [Entities: People only]
* N12: `{E1} र {E2} मध्ये {E1} अग्लो छ।` (E1 > E2)
* N14: `{E1} को उचाइ {E2} र {E3} दुवैको भन्दा बढी छ।` (E1 > E2, E1 > E3)

### Age
*Compatible Entities: People, Animals, Historical Artifacts*
*Task Type: Attribute or Direct Comparison*
* N16: `{E1} को उमेर {V1} वर्ष हो।` (Attribute: Age = V1)
* N17: `{E1}, {E2} भन्दा उमेरमा जेठो छ।` (E1 > E2)
* N18: `{E1} को उमेर {E2} को भन्दा कम छ।` (E1 < E2)
* N19: `{E1}, {E2} भन्दा उमेरमा कान्छो छ, तर {E3} त्योभन्दा पनि कान्छो छ।` (E1 < E2, E3 < E1)
* N20: `{E1} र {E2} को उमेर बराबर छ।` (E1 = E2)
* N22: `{E1} को उमेर {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।` (E1 > E2, E1 < E3)
* N23: `{E1} र {E2} को उमेर समान छ, जबकि {E3} दुवैभन्दा जेठो छ।` (E1 = E2, E3 > E1, E3 > E2)

### Price
*Compatible Entities: Products, Books, Vehicles, Properties*
*Task Type: Attribute or Direct Comparison*
* N24: `{E1} को मूल्य {V1} रुपैयाँ छ।` (Attribute: Price = V1)
* N25: `{E1}, {E2} भन्दा महँगो छ।` (E1 > E2)
* N26: `{E1} को मूल्य {E2} को भन्दा कम छ।` (E1 < E2)
* N27: `{E1} र {E2} मध्ये {E1} बढी महँगो छ।` (E1 > E2)
* N29: `{E1} को मूल्य {E2} र {E3} दुवैको भन्दा बढी छ।` (E1 > E2, E1 > E3)
* N30: `{E1} को मूल्य {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।` (E1 > E2, E1 < E3)
* N31: `{E1} र {E2} को मूल्य समान छ, जबकि {E3} को मूल्य दुवैको भन्दा बढी छ।` (E1 = E2, E3 > E1, E3 > E2)

### Quantity / Counts
*Compatible Entities: Collections of items, countable objects*
*Task Type: Attribute or Direct Comparison*
* N32: `{E1} सँग {V1} वटा किताब छन्।` (Attribute: Count = V1)
* N33: `{E1} सँग {E2} को भन्दा धेरै किताब छन्।` (E1 > E2)
* N34: `{E1} सँग {E2} को भन्दा कम सामान छ।` (E1 < E2)
* N35: `{E1} मा {E2} को तुलनामा बढी मात्रा छ।` (E1 > E2) [Entities: Uncountable substances]
* N36: `{E1} को संख्या {E2} को भन्दा बढी छ।` (E1 > E2)
* N37: `{E1} र {E2} को संख्या बराबर छ।` (E1 = E2)
* N38: `{E1} को मात्रा {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।` (E1 > E2, E1 < E3)

### Transitive / Ordering
*Compatible Entities: Abstract Concepts, Ranked Items, People*
*Task Type: Transitive Reasoning*
* N40: `{E1}, {E2} भन्दा ठूलो छ र {E2}, {E3} भन्दा ठूलो छ।` (E1 > E2, E2 > E3)
* N41: `{E1}, {E2} भन्दा सानो छ, जबकि {E3}, {E2} भन्दा ठूलो छ।` (E1 < E2, E3 > E2)
* N42: `{E1}, {E2} भन्दा अगाडि छ र {E3}, {E1} भन्दा अगाडि छ।` (E1 > E2, E3 > E1)
* N43: `{E1}, {E2} भन्दा माथि छ र {E3}, {E2} भन्दा तल छ।` (E1 > E2, E3 < E2)
* N44: `{E1} र {E2} बराबर छन्, तर {E3} दुवैभन्दा बढी छ।` (E1 = E2, E3 > E1)
* N45: `{E1}, {E2} भन्दा बढी छ र {E3}, {E1} भन्दा कम छ।` (E1 > E2, E3 < E1)

---
## Audit Notes & Rejected Templates
- **H8, N8:** Rejected due to conflicting solver mapping (equality alongside max/min questions not natively supported by direct comparison solver).
- **H13, H15, H21, H28, H39 (and Nepali equivalents):** Rejected because they contain superlatives ("सबसे लंबा है", "सबैभन्दा महँगो वस्तु हो") that give away the answer trivially in the premise text.
- **H23, H31, H45 (and Nepali equivalents):** Approved with modification to replace the Chinese full-stop `。` with the Devanagari Danda `।` to prevent `<unk>` tokens.
- **H43, N43:** Approved with modification to replace the superlative ("सबसे ऊपर है") with a relative mapping ("{E1}, {E2} से ऊपर है") to prevent giving away the answer trivially.
