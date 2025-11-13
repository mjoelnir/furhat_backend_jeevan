package furhatos.app.templateadvancedskill.language

import furhatos.flow.kotlin.FlowControlRunner
import furhatos.flow.kotlin.furhat
import furhatos.util.Language

/**
 * Tracks the current language of the robot (English or Norwegian).
 */
object LanguageManager {
    var current: AppLanguage = AppLanguage.EN
}

/**
 * Extension function that can be called from any Furhat state (because "this" is a FlowControlRunner).
 *
 * Sets Furhat's:
 *   - ASR language
 *   - TTS voice
 *
 * Usage inside a state:
 *     setAppLanguage(AppLanguage.NO)
 */
fun FlowControlRunner.setAppLanguage(lang: AppLanguage) {
    if (LanguageManager.current == lang) return  // No change needed
    LanguageManager.current = lang

    when (lang) {
        AppLanguage.EN ->
            furhat.setVoice(Language.ENGLISH_US, "Matthew", true)

        AppLanguage.NO ->
            furhat.setVoice(Language.NORWEGIAN, "Hans", true)
    }
}