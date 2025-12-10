package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.perception.PerceptionClient
import furhatos.records.User
import java.util.concurrent.ConcurrentHashMap

/**
 * Simple per-user registry so we can hang a PerceptionClient off a User record
 * without modifying the Furhat SDK types. Keeps mapping in memory only.
 */
private object PerceptionClientRegistry {
    private val clients = ConcurrentHashMap<String, PerceptionClient>()

    fun get(userId: String): PerceptionClient? = clients[userId]

    fun set(userId: String, client: PerceptionClient?) {
        if (client == null) {
            clients.remove(userId)
        } else {
            clients[userId] = client
        }
    }
}

/**
 * Extension property to access a per-user PerceptionClient.
 */
var User.perceptionClient: PerceptionClient?
    get() = PerceptionClientRegistry.get(id)
    set(value) {
        PerceptionClientRegistry.set(id, value)
    }