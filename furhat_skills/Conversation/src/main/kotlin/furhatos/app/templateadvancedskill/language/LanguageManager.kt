package furhatos.app.templateadvancedskill.language

import furhatos.flow.kotlin.FlowControlRunner
import furhatos.flow.kotlin.furhat
import furhatos.util.Language

/**
 * Tracks the current language of the robot (English or Norwegian).
 * `pinned` is set when the user explicitly asks for a language and prevents
 * auto-detection overrides until cleared.
 */
object LanguageManager {
    var current: AppLanguage = AppLanguage.EN
    var pinned: AppLanguage? = null
}

/**
 * Extension to set Furhat ASR + TTS for the given language.
 *
 * Uses built-in Polly voices:
 *  - EN: Kendra-Neural (fallback to default EN if not installed)
 *  - NO: Ida-Neural (fallback to default NO if not installed)
 *
 * Called from any state (FlowControlRunner receiver).
 */
fun FlowControlRunner.setAppLanguage(lang: AppLanguage) {
    if (LanguageManager.current == lang) return  // No change needed
    LanguageManager.current = lang

    when (lang) {
        AppLanguage.EN -> {
            furhat.setInputLanguage(Language.ENGLISH_US)
            runCatching { furhat.setVoice(language = Language.ENGLISH_US, name = "Kendra-Neural", setInputLanguage = true) }
                .recoverCatching { furhat.setVoice(language = Language.ENGLISH_US, setInputLanguage = true) }
        }

        AppLanguage.NO -> {
            furhat.setInputLanguage(Language.NORWEGIAN)
            runCatching { furhat.setVoice(language = Language.NORWEGIAN, name = "Ida-Neural", setInputLanguage = true) }
                .recoverCatching { furhat.setVoice(language = Language.NORWEGIAN, setInputLanguage = true) }
        }
    }
}