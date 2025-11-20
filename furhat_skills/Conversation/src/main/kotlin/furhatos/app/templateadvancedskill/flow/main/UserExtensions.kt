package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.perception.PerceptionClient
import furhatos.records.User
import java.util.concurrent.ConcurrentHashMap

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

var User.perceptionClient: PerceptionClient?
    get() = PerceptionClientRegistry.get(id)
    set(value) {
        PerceptionClientRegistry.set(id, value)
    }