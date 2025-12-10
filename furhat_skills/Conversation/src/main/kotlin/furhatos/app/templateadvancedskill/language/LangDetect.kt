package furhatos.app.templateadvancedskill.language

/**
 * Lightweight heuristic EN/NO detector tuned for short ASR snippets.
 * Avoids extra dependencies; relies on diacritics, stopwords, suffixes,
 * and simple n-gram hints to bias toward Norwegian when signals are clear.
 */
object LangDetect {
    private var lastStable: AppLanguage = AppLanguage.EN

    /**
     * Heuristic detector for English vs Norwegian.
     *
     * Design choices (no external packages to keep footprint small):
     *  - Immediate NO if Norwegian diacritics (æ/ø/å) are present.
     *  - Token scoring against compact stopword lists for EN/NO.
     *  - Norwegian boosts via suffix patterns, common bigrams, and “kj/skj” chars.
     *  - Margin-based decision to avoid flip-flopping; otherwise reuse lastStable.
     *
     * Tuned for short ASR hypotheses where statistical detectors often overfit to English.
     */
    fun detect(text: String): AppLanguage {
        val normalized = text.lowercase()
            .replace(Regex("[^\\p{L}\\p{M}\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

        if (normalized.isBlank()) return lastStable

        // Fast path: Norwegian diacritics
        if (normalized.any { it == 'æ' || it == 'ø' || it == 'å' }) {
            lastStable = AppLanguage.NO
            return AppLanguage.NO
        }

        val tokens = normalized.split(" ").filter { it.isNotBlank() }

        // Minimal stopword sets chosen for low false positives on short utterances
        val noStops = setOf(
            "og", "i", "på", "ikke", "hva", "hvorfor", "hvordan", "hvor", "hvem", "hvilken", "hvilket",
            "den", "det", "dette", "der", "til", "skal", "kan", "vil", "bare", "litt", "omtrent",
            "snakk", "norsk", "engelsk", "quiz", "spørsmål", "svaret", "riktig", "feil",
            "nei", "ja", "gjerne", "takk", "vær", "vær så snill", "vennligst", "flott", "supert"
        )
        val enStops = setOf(
            "and", "in", "on", "the", "a", "of", "why", "how", "what", "where", "who",
            "this", "that", "there", "to", "will", "can", "just", "little", "about",
            "speak", "english", "norwegian", "quiz", "question", "answer", "right", "wrong"
        )

        // Explicit language keywords override weak signals
        if (tokens.any { it == "norsk" || it == "norwegian" || it == "på" && tokens.contains("norsk") }) {
            lastStable = AppLanguage.NO
            return AppLanguage.NO
        }
        if (tokens.any { it == "english" || it == "engelsk" }) {
            lastStable = AppLanguage.EN
            return AppLanguage.EN
        }

        var noScore = 0
        var enScore = 0

        for (t in tokens) {
            if (t in noStops) noScore++
            if (t in enStops) enScore++
        }

        // Boost Norwegian if any token ends with common Norwegian suffixes
        val noSuffixes = listOf("en", "et", "ene", "ende", "ene", "het", "heten", "lig", "lige")
        if (tokens.any { t -> noSuffixes.any { suf -> t.length > suf.length && t.endsWith(suf) } }) {
            noScore += 2
        }

        // Bigram hints (common Norwegian question openers)
        val textNoPunct = normalized
        if (textNoPunct.startsWith("hva er") || textNoPunct.startsWith("hvordan") || textNoPunct.startsWith("hvorfor") || textNoPunct.startsWith("hvor er")) {
            noScore += 2
        }

        // Character n-gram hints (simple heuristic)
        if (normalized.contains("kj") || normalized.contains("skj")) {
            noScore += 1
        }

        // Decide by margin; if close, use last stable to prevent flip-flop.
        val decision = when {
            noScore >= enScore + 2 -> AppLanguage.NO
            enScore >= noScore + 2 -> AppLanguage.EN
            else -> lastStable
        }
        lastStable = decision
        return decision
    }
}