package furhatos.app.templateadvancedskill.language

object LangDetect {
    fun detect(text: String): AppLanguage {
        val t = text.lowercase()

        // Check for Norwegian special characters
        val norwegianChars = listOf('æ', 'ø', 'å')
        if (norwegianChars.any { it in t }) {
            return AppLanguage.NO
        }

        // Common Norwegian words
        val norwegianWords = listOf("ikke", "hvordan", "hva", "hvorfor", "forklar", "beskriv", "omtrent")
        if (norwegianWords.any { t.contains(it) }) {
            return AppLanguage.NO
        }

        // Default: English
        return AppLanguage.EN
    }
}