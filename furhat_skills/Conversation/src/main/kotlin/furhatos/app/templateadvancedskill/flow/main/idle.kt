package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.State
import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.onUserEnter
import furhatos.flow.kotlin.state
import furhatos.app.templateadvancedskill.flow.main.Greeting

val Idle: State = state {
    onEntry {
        furhat.attendNobody()
    }

    onUserEnter {
        furhat.attend(it)
        goto(Greeting)
    }

}