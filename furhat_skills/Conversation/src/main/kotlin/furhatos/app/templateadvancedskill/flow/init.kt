package furhatos.app.templateadvancedskill.flow

import furhatos.app.templateadvancedskill.flow.main.DocumentWaitingToStart
import furhatos.app.templateadvancedskill.flow.main.Greeting
import furhatos.app.templateadvancedskill.flow.main.Idle
import furhatos.app.templateadvancedskill.setting.*
import furhatos.flow.kotlin.State
import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.state
import furhatos.flow.kotlin.users
import furhatos.app.templateadvancedskill.setting.DISTANCE_TO_ENGAGE

val Init: State = state {
    init {
        /** Set our default interaction parameters */
        users.setSimpleEngagementPolicy(DISTANCE_TO_ENGAGE, MAX_NUMBER_OF_USERS)
        // Add a small end-of-speech silence buffer so the robot waits
        // ~1.5–2 seconds after the user stops talking before responding.
        // This helps avoid interrupting users who pause briefly mid-utterance.
        furhat.param.endSilTimeout = 2000
    }
    onEntry {
        /** start interaction */
        when {
            furhat.isVirtual() -> goto(Greeting) // Convenient to bypass the need for user when running Virtual Furhat
            users.hasAny() -> {
                furhat.attend(users.random)
                goto(DocumentWaitingToStart)
            }
            else -> goto(Idle)
        }
    }

}