package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.language.AppLanguage
import furhatos.app.templateadvancedskill.language.I18n
import furhatos.app.templateadvancedskill.language.setAppLanguage
import furhatos.flow.kotlin.*

val Greeting : State = state {

    onEntry {
        // Default ASR/TTS to English at startup
        setAppLanguage(AppLanguage.EN)

        // Bilingual intro (English + Norwegian)
        furhat.say(I18n.t("intro_bilingual"))

        // If you have a specific document/topic, plug its name here
        val docName = "this topic"
        furhat.say(I18n.t("greet", docName))

        goto(DocumentWaitingToStart)
    }
}