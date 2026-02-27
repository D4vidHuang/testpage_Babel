from .prompt_base import PromptBase


class PromptLibrary(PromptBase):
    """Szablony promptów w języku polskim dla ewaluacji LLM-as-Judge.

    Dziedziczy narzędzia niezależne od języka z PromptBase.
    Tutaj zdefiniowana jest tylko treść promptów specyficznych dla języka polskiego.
    """
    
    LANGUAGE_CODE = "polish"
    
    @staticmethod
    def get_output_field_names() -> dict:
        """Polish translations for JSON output field names."""
        return {
            "evaluations": "ewaluacje",
            "model_name": "nazwa_modelu",
            "errors": "bledy",
            "error_id": "id_bledu",
            "confidence": "pewnosc",
            "justification": "uzasadnienie",
            "explanation": "wyjasnienie",
            "reasoning": "rozumowanie",
            "overall_quality": "ogolna_jakosc",
            "correct": "poprawne",
            "partially_correct": "czesciowo_poprawne",
            "incorrect": "niepoprawne",
        }

    @staticmethod
    def get_rubric_labels() -> dict:
        """Polish translations for rubric formatting labels."""
        return {
            "rubric_title": "Rubryka Klasyfikacji Błędów",
            "definition": "Definicja",
            "mark_present": "Oznacz jako OBECNY jeśli",
            "mark_absent": "Oznacz jako NIEOBECNY jeśli",
        }

    @staticmethod
    def system_basic() -> str:
        """Generuje podstawowy prompt systemowy dla roli ewaluatora."""
        return (
            "Jesteś ekspertem oceniającym. Analizujesz kod Java, komentarze "
            "i dokumentację pod kątem poprawności na podstawie podanej taksonomii błędów.\\n\\n"
            "Instrukcje łagodzenia uprzedzeń:\\n"
            "- Unikaj uprzedzenia do rozwlekłości (Verbosity Bias): Nie faworyzuj dłuższych komentarzy. Oceniaj na podstawie poprawności.\\n"
            "- Zapewnij spójność: Stosuj te same standardy do wszystkich predykcji.\\n"
            "- Zapobiegaj systematycznym błędom: Oceniaj każdy przypadek niezależnie w oparciu o taksonomię."
        )

    @staticmethod
    def system_enhanced() -> str:
        """Generuje rozszerzony prompt systemowy z łagodzeniem uprzedzeń."""
        return (
            "Jesteś ekspertem w dziedzinie oceny dokumentacji kodu z doświadczeniem w "
            "inżynierii oprogramowania i wielojęzycznej analizie tekstu. Twoim zadaniem jest "
            "identyfikacja błędów w komentarzach wygenerowanych przez LLM przy użyciu szczegółowej taksonomii.\\n\\n"

            "Zasady oceny:\n"
            "- Bądź spójny: Stosuj te same standardy niezależnie od długości lub rozwlekłości komentarza\n"
            "- Bądź precyzyjny: Zgłaszaj błędy tylko wtedy, gdy kryteria włączenia są wyraźnie spełnione\n"
            "- Sprawdzaj włączenia/wyłączenia: Zawsze weryfikuj kryteria włączenia/wyłączenia przed potwierdzeniem błędu\n"
            "- Świadomość językowa: Pamiętaj, że komentarze mogą być w językach innych niż angielski; oceniaj gramatykę i semantykę odpowiednio dla języka docelowego\n"
            "- Zależność od kontekstu: Używaj otaczającego kodu do oceny dokładności semantycznej\n"
            "- Niezależność od pozycji: Oceniaj każdą predykcję niezależnie; nie pozwól, aby kolejność wpływała na Twoją ocenę\n\n"

            "WAŻNE: Nie pozwól, aby długość komentarza wpływała na Twoją ocenę jakości. "
            "Krótkie komentarze mogą być poprawne; długie komentarze mogą zawierać błędy. "
            "Oceniaj każdy komentarz wyłącznie na podstawie tego, czy dokładnie opisuje kod "
            "zgodnie z kryteriami taksonomii."
        )

    @staticmethod
    def expert_accuracy_guidance() -> str:
        return (
            "Etykiety dokładności eksperckiej:\n"
            "- poprawne: Przewidziany komentarz zawiera wszystkie informacje z oryginalnego komentarza; dodatkowe istotne szczegóły są dopuszczalne.\n"
            "- czesciowo_poprawne: Komentarz jest w większości prawidłowy, ale zawiera drobny błąd (np. literówkę, złą nazwę zmiennej lub niewielkie pominięcie).\n"
            "- niepoprawne: Komentarz pomija ważne informacje lub zawiera istotne błędy wykraczające poza drobne poprawki."
        )

    @staticmethod
    def assignment_evaluate_models(taxonomy: str) -> str:
        """Generuje prompt przydziału zadania do oceny pogrupowanych predykcji modeli."""
        return (
            "Otrzymujesz kod Java z oryginalnym komentarzem i wieloma predykcjami wygenerowanymi przez modele w formacie JSON. "
            "Twoim zadaniem jest ocena komentarza przewidzianego przez każdy model i identyfikacja błędów z taksonomii.\n\n"
            f"Taksonomia możliwych błędów w komentarzach:\n{taxonomy}\n\n"
            "Dla każdej predykcji modelu w dostarczonym formacie JSON:\n"
            "- Przeanalizuj pole przewidziany_komentarz.\n"
            "- Zdecyduj, które typy błędów z taksonomii są obecne.\n"
            "- Przy podejmowaniu decyzji odwołuj się do kryteriów włączenia i wyłączenia.\n"
            "- Jeśli nie ma żadnych błędów, wskaż to wyraźnie.\n\n"
            "Zasady oceny:\n"
            "- Bądź spójny: Stosuj te same standardy niezależnie od długości lub rozwlekłości komentarza\n"
            "- Bądź precyzyjny: Zgłaszaj błędy tylko wtedy, gdy kryteria włączenia są wyraźnie spełnione\n"
            "- Sprawdzaj włączenia/wyłączenia: Zawsze weryfikuj kryteria włączenia/wyłączenia przed potwierdzeniem błędu\n"
            "- Świadomość językowa: Pamiętaj, że komentarze mogą być w językach innych niż angielski; oceniaj gramatykę i semantykę odpowiednio dla języka docelowego\n"
            "- Zależność od kontekstu: Używaj otaczającego kodu do oceny dokładności semantycznej\n"
            "- Niezależność od pozycji: Oceniaj każdą predykcję niezależnie; nie pozwól, aby kolejność wpływała na Twoją ocenę\n\n"
            "Dane wejściowe są dostarczane jako JSON z polami source_code, original_comment oraz tablicą model_predictions."
        )

    @staticmethod
    def output_basic() -> str:
        """Generuje instrukcje dla ewaluacji danych grupowych z wieloma modelami."""
        return (
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Zwróć ustrukturyzowany JSON z wynikami dla każdego modelu:\n"
            "{\n"
            '  "ewaluacje": [\n'
            "    {\n"
            '      "nazwa_modelu": "<identyfikator_modelu>",\n'
            '      "bledy": ["<id_bledu>", ...],\n'
            '      "wyjasnienie": "Krótkie wyjaśnienie znalezionych błędów.",\n'
            '      "ogolna_jakosc": "<poprawne|czesciowo_poprawne|niepoprawne>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Jeśli model nie ma błędów, użyj pustej tablicy błędów."
        )

    @staticmethod
    def output_with_reasoning() -> str:
        """Ewaluacja chain-of-thought z jawnymi krokami rozumowania."""
        return (
            "Dla przewidzianego komentarza każdego modelu wykonaj te kroki:\\n"
            "1. Przeczytaj uważnie przewidziany komentarz\\n"
            "2. Zrozum, do której części kodu odnosi się wygenerowany komentarz, bazując na technice maskowania Fill-in-the-Middle (FIM)\\n"
            "3. Porównaj go z oryginalnym komentarzem i kontekstem kodu\\n"
            "4. Dla każdego potencjalnego błędu zastanów się, czy kryteria włączenia są spełnione\\n"
            "5. Sprawdź kryteria wyłączenia przed potwierdzeniem błędu\\n"
            "6. Podaj swoje rozumowanie, a następnie klasyfikację\\n"
            "7. Na podstawie powyższego rozumowania określ ogólną jakość wygenerowanego komentarza\\n\\n"
            "Zasady oceny:\\n"
            "- Bądź spójny: Stosuj te same standardy niezależnie od długości lub rozwlekłości komentarza\\n"
            "- Bądź precyzyjny: Zgłaszaj błędy tylko wtedy, gdy kryteria włączenia są wyraźnie spełnione\\n"
            "- Sprawdzaj włączenia/wyłączenia: Zawsze weryfikuj kryteria włączenia/wyłączenia przed potwierdzeniem błędu\\n"
            "- Świadomość językowa: Pamiętaj, że komentarze mogą być w językach innych niż angielski; oceniaj gramatykę i semantykę odpowiednio dla języka docelowego\\n"
            "- Zależność od kontekstu: Używaj otaczającego kodu do oceny dokładności semantycznej\\n"
            "- Niezależność od pozycji: Oceniaj każdą predykcję niezależnie; nie pozwól, aby kolejność wpływała na Twoją ocenę\\n\\n"
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Zwróć ustrukturyzowany JSON z wynikami dla każdego modelu:\\n"
            "{\\n"
            '  "ewaluacje": [\\n'
            "    {\\n"
            '      "nazwa_modelu": "<identyfikator_modelu>",\\n'
            '      "rozumowanie": "Analiza krok po kroku: Po pierwsze zauważam, że...",\\n'
            '      "bledy": [\\n'
            "        {\\n"
            '          "id_bledu": "<id_bledu>",\\n'
            '          "pewnosc": <0.0-1.0>,\\n'
            '          "uzasadnienie": "Ten błąd ma zastosowanie, ponieważ..."\\n'
            "        }\\n"
            "      ],\\n"
            '      "ogolna_jakosc": "<poprawne|czesciowo_poprawne|niepoprawne>"\\n'
            "    }\\n"
            "  ]\\n"
            "}\\n"
            "Jeśli model nie ma błędów, zwróć pustą tablicę błędów wraz z rozumowaniem "
            "wyjaśniającym, dlaczego komentarz jest poprawny."
        )

    @staticmethod
    def assignment_evaluate_cluster(cluster_name: str, cluster_taxonomy: str) -> str:
        """Generuje prompt przydziału zadania dla konkretnego klastra kategorii."""
        cluster_descriptions = {
            "linguistic_grammar": "poprawność językową i gramatyczną",
            "linguistic_language": "spójność językową i właściwość",
            "semantic_accuracy": "dokładność semantyczną i kompletność",
            "semantic_code": "dokładność fragmentów kodu",
            "model_behavior": "błędy specyficzne dla modelu i artefakty",
            "syntax_format": "format i strukturę komentarza",
            "meta": "kryteria wyłączenia i różne problemy"
        }
    
        description = cluster_descriptions.get(cluster_name, cluster_name)
    
        return (
            f"Oceniasz komentarze do kodu WYŁĄCZNIE pod kątem: {description.upper()}.\\n\\n"
            f"Istotne kategorie błędów dla tej oceny:\\n{cluster_taxonomy}\\n\\n"
            "WAŻNE: Skup się TYLKO na typach błędów wymienionych powyżej. "
            "Ignoruj inne potencjalne problemy — będą oceniane osobno.\\n\\n"
            "Dla każdej kategorii błędu powyżej określ, czy błąd jest obecny na podstawie "
            "kryteriów włączenia. Zawsze sprawdzaj kryteria wyłączenia przed potwierdzeniem."
        )
