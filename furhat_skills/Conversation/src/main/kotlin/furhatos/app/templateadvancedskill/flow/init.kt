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