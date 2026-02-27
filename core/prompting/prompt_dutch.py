from .prompt_base import PromptBase


class PromptLibrary(PromptBase):
    """Prompt-sjablonen in het Nederlands voor LLM-as-Judge-evaluatie.

    Erft taal-onafhankelijke hulpmiddelen van PromptBase.
    Hier is uitsluitend de promptinhoud gedefinieerd die specifiek is voor het Nederlands.
    """
    
    LANGUAGE_CODE = "dutch"
    
    @staticmethod
    def get_output_field_names() -> dict:
        """Nederlandse vertalingen voor JSON-uitvoerveldnamen."""
        return {
            "evaluations": "evaluaties",
            "model_name": "modelnaam",
            "errors": "fouten",
            "error_id": "fout_id",
            "confidence": "zekerheid",
            "justification": "rechtvaardiging",
            "explanation": "uitleg",
            "reasoning": "redenering",
            "overall_quality": "algemene_kwaliteit",
            "correct": "correct",
            "partially_correct": "gedeeltelijk_correct",
            "incorrect": "onjuist",
        }

    @staticmethod
    def get_rubric_labels() -> dict:
        """Nederlandse vertalingen voor rubriceringslabels."""
        return {
            "rubric_title": "Rubriek voor Foutclassificatie",
            "definition": "Definitie",
            "mark_present": "Markeer als AANWEZIG indien",
            "mark_absent": "Markeer als AFWEZIG indien",
        }

    @staticmethod
    def system_basic() -> str:
        """Genereert de basis-systeemprompt voor de rol van evaluator."""
        return (
            "Je bent een deskundige evaluator. Je analyseert Java-code, commentaren "
            "en documentatie op correctheid op basis van een opgegeven foutentaxonomie.\\n\\n"
            "Instructies voor bias-mitigatie:\\n"
            "- Vermijd verbositeitsbias: Geef geen voorkeur aan langere commentaren. Beoordeel op basis van correctheid.\\n"
            "- Zorg voor consistentie: Pas dezelfde standaarden toe op alle voorspellingen.\\n"
            "- Voorkom systematische fouten: Beoordeel elk geval onafhankelijk op basis van de taxonomie."
        )

    @staticmethod
    def system_enhanced() -> str:
        """Genereert een uitgebreide systeemprompt met bias-mitigatie."""
        return (
            "Je bent een expert in de evaluatie van codedocumentatie met ervaring in "
            "software-engineering en meertalige tekstanalyse. Je taak is het "
            "identificeren van fouten in door LLM gegenereerde codecommentaren met behulp van een gedetailleerde taxonomie.\\n\\n"

            "Evaluatieprincipes:\n"
            "- Wees consistent: Pas dezelfde standaarden toe ongeacht de lengte of verbositeit van het commentaar\n"
            "- Wees precies: Meld alleen fouten wanneer duidelijk aan inclusiecriteria is voldaan\n"
            "- Controleer inclusie/exclusie: Verifieer altijd inclusie- en exclusiecriteria voordat je een fout bevestigt\n"
            "- Taalsensitief: Houd er rekening mee dat commentaren in andere talen dan het Engels kunnen zijn; beoordeel grammatica en semantiek passend bij de doeltaal\n"
            "- Contextafhankelijk: Gebruik de omliggende code om semantische nauwkeurigheid te beoordelen\n"
            "- Positie-onafhankelijk: Beoordeel elke voorspelling onafhankelijk; laat de volgorde je oordeel niet beïnvloeden\n\n"

            "BELANGRIJK: Laat de lengte van een commentaar je oordeel over de kwaliteit niet beïnvloeden. "
            "Korte commentaren kunnen correct zijn; lange commentaren kunnen fouten bevatten. "
            "Beoordeel elk commentaar uitsluitend op basis van of het de code nauwkeurig beschrijft "
            "volgens de taxonomiecriteria."
        )

    @staticmethod
    def expert_accuracy_guidance() -> str:
        return (
            "Labels voor expertnauwkeurigheid:\n"
            "- correct: Het voorspelde commentaar bevat alle informatie uit het oorspronkelijke commentaar; extra relevante details zijn toegestaan.\n"
            "- gedeeltelijk_correct: Het commentaar is grotendeels correct maar bevat een kleine fout (bijv. een typefout, verkeerde variabelenaam of lichte weglating).\n"
            "- onjuist: Het commentaar mist belangrijke informatie of bevat substantiële fouten die verder gaan dan kleine correcties."
        )

    @staticmethod
    def assignment_evaluate_models(taxonomy: str) -> str:
        """Genereert een taakprompt voor het evalueren van gegroepeerde modelvoorspellingen."""
        return (
            "Je ontvangt Java-code met een oorspronkelijk commentaar en meerdere door modellen gegenereerde voorspellingen in JSON-formaat. "
            "Je taak is om het voorspelde commentaar van elk model te evalueren en fouten uit de taxonomie te identificeren.\n\n"
            f"Taxonomie van mogelijke commentaarfouten:\n{taxonomy}\n\n"
            "Voor elke modelvoorspelling in de aangeleverde JSON:\n"
            "- Analyseer het veld predicted_comment.\n"
            "- Bepaal welke fouttypes uit de taxonomie aanwezig zijn.\n"
            "- Raadpleeg inclusie- en exclusiecriteria bij het nemen van beslissingen.\n"
            "- Geef expliciet aan als er geen fouten aanwezig zijn.\n\n"
            "Evaluatieprincipes:\n"
            "- Wees consistent: Pas dezelfde standaarden toe ongeacht de lengte of verbositeit van het commentaar\n"
            "- Wees precies: Meld alleen fouten wanneer duidelijk aan inclusiecriteria is voldaan\n"
            "- Controleer inclusie/exclusie: Verifieer altijd inclusie- en exclusiecriteria voordat je een fout bevestigt\n"
            "- Taalsensitief: Houd er rekening mee dat commentaren in andere talen dan het Engels kunnen zijn; beoordeel grammatica en semantiek passend bij de doeltaal\n"
            "- Contextafhankelijk: Gebruik de omliggende code om semantische nauwkeurigheid te beoordelen\n"
            "- Positie-onafhankelijk: Beoordeel elke voorspelling onafhankelijk; laat de volgorde je oordeel niet beïnvloeden\n\n"
            "De invoer wordt aangeleverd als JSON met de velden source_code, original_comment en een model_predictions-array."
        )

    @staticmethod
    def output_basic() -> str:
        """Genereert instructies voor gegroepeerde evaluatie met meerdere modellen."""
        return (
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Geef gestructureerde JSON terug met resultaten voor elk model:\n"
            "{\n"
            '  "evaluaties": [\n'
            "    {\n"
            '      "modelnaam": "<model_id>",\n'
            '      "fouten": ["<fout_id>", ...],\n'
            '      "uitleg": "Korte uitleg van de gevonden fouten.",\n'
            '      "algemene_kwaliteit": "<correct|gedeeltelijk_correct|onjuist>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Als een model geen fouten heeft, gebruik dan een lege foutenlijst."
        )

    @staticmethod
    def output_with_reasoning() -> str:
        """Chain-of-thought-evaluatie met expliciete redeneerstappen."""
        return (
            "Voer voor het voorspelde commentaar van elk model de volgende stappen uit:\\n"
            "1. Lees het voorspelde commentaar zorgvuldig\\n"
            "2. Begrijp op welk deel van de code het gegenereerde commentaar betrekking heeft, gebaseerd op de Fill-in-the-Middle (FIM)-maskeringstechniek\\n"
            "3. Vergelijk het met het oorspronkelijke commentaar en de codecontext\\n"
            "4. Overweeg voor elke potentiële fout of aan de inclusiecriteria is voldaan\\n"
            "5. Controleer exclusiecriteria voordat je een fout bevestigt\\n"
            "6. Geef je redenering en vervolgens je classificatie\\n"
            "7. Bepaal op basis van bovenstaande redenering de algemene kwaliteit van het gegenereerde commentaar\\n\\n"
            "Evaluatieprincipes:\\n"
            "- Wees consistent: Pas dezelfde standaarden toe ongeacht de lengte of verbositeit van het commentaar\\n"
            "- Wees precies: Meld alleen fouten wanneer duidelijk aan inclusiecriteria is voldaan\\n"
            "- Controleer inclusie/exclusie: Verifieer altijd inclusie- en exclusiecriteria voordat je een fout bevestigt\\n"
            "- Taalsensitief: Houd er rekening mee dat commentaren in andere talen dan het Engels kunnen zijn; beoordeel grammatica en semantiek passend bij de doeltaal\\n"
            "- Contextafhankelijk: Gebruik de omliggende code om semantische nauwkeurigheid te beoordelen\\n"
            "- Positie-onafhankelijk: Beoordeel elke voorspelling onafhankelijk; laat de volgorde je oordeel niet beïnvloeden\\n\\n"
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Geef gestructureerde JSON terug met resultaten voor elk model:\\n"
            "{\\n"
            '  "evaluaties": [\\n'
            "    {\\n"
            '      "modelnaam": "<model_id>",\\n'
            '      "redenering": "Stapsgewijze analyse: Eerst merk ik op dat...",\\n'
            '      "fouten": [\\n'
            "        {\\n"
            '          "fout_id": "<fout_id>",\\n'
            '          "zekerheid": <0.0-1.0>,\\n'
            '          "rechtvaardiging": "Deze fout is van toepassing omdat..."\\n'
            "        }\\n"
            "      ],\\n"
            '      "algemene_kwaliteit": "<correct|gedeeltelijk_correct|onjuist>"\\n'
            "    }\\n"
            "  ]\\n"
            "}\\n"
            "Als een model geen fouten heeft, geef dan een lege foutenlijst terug met een redenering "
            "die uitlegt waarom het commentaar correct is."
        )

    @staticmethod
    def assignment_evaluate_cluster(cluster_name: str, cluster_taxonomy: str) -> str:
        """Genereert een taakprompt voor een specifieke categoriecluster."""
        cluster_descriptions = {
            "linguistic_grammar": "taalkundige en grammaticale correctheid",
            "linguistic_language": "taalconsistentie en geschiktheid",
            "semantic_accuracy": "semantische nauwkeurigheid en volledigheid",
            "semantic_code": "nauwkeurigheid van codefragmenten",
            "model_behavior": "modelspecifieke fouten en artefacten",
            "syntax_format": "commentaarformaat en structuur",
            "meta": "exclusiecriteria en diverse kwesties"
        }
    
        description = cluster_descriptions.get(cluster_name, cluster_name)
    
        return (
            f"Je evalueert codecommentaren UITSLUITEND op: {description.upper()}.\\n\\n"
            f"Relevante foutcategorieën voor deze evaluatie:\\n{cluster_taxonomy}\\n\\n"
            "BELANGRIJK: Richt je ALLEEN op de hierboven vermelde fouttypes. "
            "Negeer andere potentiële problemen — deze worden afzonderlijk geëvalueerd.\\n\\n"
            "Bepaal voor elke bovenstaande foutcategorie of de fout aanwezig is op basis van "
            "de inclusiecriteria. Controleer altijd de exclusiecriteria voordat je bevestigt."
        )
