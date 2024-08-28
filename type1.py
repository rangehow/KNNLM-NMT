prefix = "Translate this from Deutsch to English:"
shot = {
    "it": """\nDeutsch:Zeigt den aktuellen Wert der Feldvariable an.\nEnglish:Displays the current value of the field variable.\nDeutsch:In diesem Bereich wählen Sie die relativen Größen bezogen auf die Basisgröße.\nEnglish:In this section, you can determine the relative sizes for each type of element with reference to the base size.\nDeutsch:Geben Sie einen kurzen, beschreibenden Namen für die Schnittstelle ein.\nEnglish:Simply enter a short human-readable description for this device.""",
    "koran": """\nDeutsch:So führt Gott (im Gleichnis) das Wahre und das Falsche an.\nEnglish:This is how God determines truth and falsehood.\nDeutsch:Da kamen sie auf ihn zu geeilt.\nEnglish:So the people descended upon him.\nDeutsch:Wir begehren von euch weder Lohn noch Dank dafür.\nEnglish:We wish for no reward, nor thanks from you.""",
    "law": """\nDeutsch:Deshalb ist die Regelung von der Ausfuhrleistung abhängig.\nEnglish:In this regard, the scheme is contingent upon export performance.\nDeutsch:Das Mitglied setzt gleichzeitig den Rat von seinem Beschluß in Kenntnis.\nEnglish:That member shall simultaneously inform the Council of the action it has taken.\nDeutsch:Dies gilt auch für die vorgeschlagene Sicherheitsleistung.\nEnglish:The same shall apply as regards the security proposed.""",
    "medical": """\nDeutsch:Das Virus wurde zuerst inaktiviert (abgetötet), damit es keine Erkrankungen verursachen kann.\nEnglish:This may help to protect against the disease caused by the virus.\nDeutsch:Desirudin ist ein rekombinantes DNS-Produkt, das aus Hefezellen hergestellt wird.\nEnglish:Desirudin is a recombinant DNA product derived from yeast cells.\nDeutsch:Katzen erhalten eine intramuskuläre Injektion.\nEnglish:In cats, it is given by intramuscular injection.""",
}
postfix = "\nDeutsch:{de}\nEnglish:{en}"
prompt = prefix + shot['it'] + postfix


b={'de':'hi'}
print(prompt.format_map({'en':b['en'],'de':b['de']}))
