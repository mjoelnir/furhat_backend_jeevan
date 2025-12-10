package furhatos.app.templateadvancedskill.nlu

import furhatos.nlu.Intent
import furhatos.util.Language
import furhatos.nlu.common.PersonName
import furhatos.app.templateadvancedskill.language.AppLanguage

/**
 * Define intents to match a user utterance and assign meaning to what they said.
 * Note that there are more intents available in the Asset Collection in furhat.libraries.standard.NluLib
 **/

class NiceToMeetYouIntent : Intent() {
    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.NORWEGIAN -> listOf(
                "hyggelig å møte deg",
                "godt å se deg",
                "hyggelig å treffe deg",
                "så kjekt å møte deg",
                "hyggelig å se deg",
                "fint å endelig møte deg",
                "kjekt å treffes"
            )
            else -> listOf(
                "glad to meet you",
                "a pleasure to meet you",
                "nice to see you",
                "great to meet you",
                "happy to see you",
                "very nice to finally meet you",
                "fun to meet up with you"
            )
        }
    }
}

class HowAreYouIntent : Intent() {
    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.NORWEGIAN -> listOf(
                "hvordan går det",
                "hvordan har du det",
                "hvordan går det i dag",
                "går det bra",
                "hvordan står det til",
                "hva skjer",
                "hvordan føler du deg"
            )
            else -> listOf(
                "how are you",
                "how are you doing today",
                "what's up",
                "how are things with you",
                "how's it going?",
                "how are you feeling",
                "how's life",
                "what's going on with you"
            )
        }
    }
}

class HelpIntent : Intent() {
    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.NORWEGIAN -> listOf(
                "jeg trenger hjelp",
                "kan du hjelpe meg",
                "hjelp meg",
                "kan noen hjelpe",
                "jeg trenger assistanse"
            )
            else -> listOf(
                "I need help",
                "help me please",
                "can someone help me",
                "I need assistance"
            )
        }
    }
}

class WhatIsThisIntent : Intent() {
    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.NORWEGIAN -> listOf(
                "hva er dette",
                "hva skal jeg si",
                "hva burde jeg si",
                "jeg vet ikke hva jeg skal gjøre",
                "hva gjør jeg nå",
                "burde jeg si noe",
                "hva skjer",
                "hva er det som skjer her",
                "kan noen si meg hva som foregår"
            )
            else -> listOf(
                "What is this",
                "what am I supposed to say",
                "what should I say",
                "I don't know what to do",
                "what am I supposed to do now",
                "should I say something",
                "what's going on",
                "what is happening here",
                "can someone tell me what is going on"
            )
        }
    }
}

class UncertainResponseIntent : Intent() {
    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.NORWEGIAN -> listOf(
                "jeg vet ikke",
                "jeg er ikke sikker",
                "hva tenker du",
                "hva er din mening",
                "kanskje",
                "muligens",
                "det kan være",
                "jeg er usikker",
                "jeg vet ikke helt",
                "ikke helt sikker på det",
                "jeg er ikke sikker på hva jeg skal si",
                "jeg er ikke sikker på hva jeg skal gjøre"
            )
            else -> listOf(
                "I don't know",
                "I'm not sure",
                "what do you think",
                "what's your opinion",
                "I see it being",
                "I think it could be",
                "maybe",
                "possibly",
                "perhaps",
                "I'm not certain",
                "I'm uncertain",
                "I'm not sure about that",
                "I'm not sure what to think",
                "I'm not sure what to say",
                "I'm not sure what to do",
                "I'm not sure what to make of that",
                "I'm not sure what to make of this",
                "I'm not sure what to make of it"
            )
        }
    }
}

/**
 * capture the user's name in both English and Norwegian.
 */
class MyNameIsIntent(
    val name: PersonName? = null
) : Intent() {

    override fun getExamples(lang: Language): List<String> {
        return when (lang) {
            Language.ENGLISH_US -> listOf(
                "My name is @name",
                "I am @name",
                "I'm @name",
                "You can call me @name",
                "It's @name"
            )
            Language.NORWEGIAN -> listOf(
                "Jeg heter @name",
                "Mitt navn er @name",
                "Jeg er @name",
                "Du kan kalle meg @name"
            )
            else -> listOf(
                "My name is @name",
                "I am @name"
            )
        }
    }
}