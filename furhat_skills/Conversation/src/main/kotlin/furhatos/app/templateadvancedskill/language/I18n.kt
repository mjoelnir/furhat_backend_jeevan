package furhatos.app.templateadvancedskill.language

object I18n {
    private val strings = mapOf(
        AppLanguage.EN to mapOf(
            "intro_bilingual" to "Hi! Hei! You can talk to me in English or Norwegian. Just start talking.",
            "greet" to "I’m here to help you learn about %s. What would you like to know?",
            "fallback" to "I didn’t catch that. Could you please repeat your question?",
            "error_processing" to "I encountered an error processing your request. Please try again.",
            "waiting_doc" to "I’ll start by looking at the document and then we can talk about it.",
            "goodbye" to "Thank you for the interesting conversation! Goodbye!"
        ),
        AppLanguage.NO to mapOf(
            "intro_bilingual" to "Hi! Hei! Du kan snakke med meg på engelsk eller norsk. Bare start å snakke.",
            "greet" to "Jeg er her for å hjelpe deg med %s. Hva vil du vite?",
            "fallback" to "Jeg oppfattet ikke det. Kan du gjenta spørsmålet?",
            "error_processing" to "Jeg fikk en feil da jeg behandlet forespørselen din. Prøv igjen.",
            "waiting_doc" to "Jeg starter med å se på dokumentet, så kan vi snakke om det etterpå.",
            "goodbye" to "Takk for en hyggelig samtale! Ha det bra!"
        )
    )

    fun t(key: String, vararg args: Any?): String {
        val langMap = strings[LanguageManager.current] ?: strings[AppLanguage.EN]!!
        val raw = langMap[key] ?: strings[AppLanguage.EN]!![key]!!
        return if (args.isNotEmpty()) raw.format(*args) else raw
    }
}