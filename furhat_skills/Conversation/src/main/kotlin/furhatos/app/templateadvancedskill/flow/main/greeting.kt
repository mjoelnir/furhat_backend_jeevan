package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.language.AppLanguage
import furhatos.app.templateadvancedskill.language.setAppLanguage
import furhatos.app.templateadvancedskill.params.BACKEND_URL
import furhatos.app.templateadvancedskill.perception.PerceptionClient
import furhatos.flow.kotlin.*
import furhatos.flow.kotlin.Furhat
import furhatos.flow.kotlin.furhat.audiofeed.AudioFeedListener
import furhatos.flow.kotlin.furhat.camerafeed.CameraFeedListener
import furhatos.flow.kotlin.furhat.camerafeed.FaceData
import furhatos.flow.kotlin.users
import furhatos.records.User
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.awt.image.BufferedImage

private object PerceptionStreaming {
    @Volatile
    var activeClient: PerceptionClient? = null
    @Volatile
    private var listenersAttached: Boolean = false

    fun start(client: PerceptionClient) {
        activeClient = client
    }

    fun stop() {
        activeClient = null
    }

    fun ensureListeners(furhat: Furhat) {
        if (listenersAttached) return
        furhat.cameraFeed.addListener(PerceptionCameraListener)
        furhat.audioFeed.addListener(PerceptionAudioListener)
        listenersAttached = true
    }
}

private object PerceptionCameraListener : CameraFeedListener {
    override fun cameraImage(image: BufferedImage, imageData: ByteArray, faces: List<FaceData>) {
        val client = PerceptionStreaming.activeClient ?: return
        val payload = byteArrayOf(0x01) + imageData
                println("PerceptionCameraListener sending frame bytes=${payload.size}")
        client.sendBinary(payload)
    }
}

private object PerceptionAudioListener : AudioFeedListener {
    override fun audioData(data: ByteArray) {
        val client = PerceptionStreaming.activeClient ?: return
        val payload = byteArrayOf(0x02) + data
                println("PerceptionAudioListener sending audio bytes=${payload.size}")
        client.sendBinary(payload)
    }
}

private val PERCEPTION_WS_URL: String =
    BACKEND_URL.replaceFirst("http", "ws") + "/ws/perception"
private val perceptionScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

       /**
        * Force-start (or restart) perception WS streaming for the current user/session.
        * Always sets a fresh sessionId and ensures listeners are attached.
        * Uses the user-scoped perceptionClient property to keep association stable.
        */
       fun restartPerceptionClient(furhat: Furhat, currentUser: User) {
           // Close any existing client
           currentUser.perceptionClient?.close()
           PerceptionStreaming.stop()

           val sessionId = "furhat-session-" + System.currentTimeMillis()
           val client = PerceptionClient(
               backendWsUrl = PERCEPTION_WS_URL,
               sessionId = sessionId,
               robotId = "furhat-ntnu-01"
           )

           currentUser.perceptionClient = client
           client.connect(perceptionScope)
           PerceptionStreaming.start(client)
           PerceptionStreaming.ensureListeners(furhat)
           println("Perception WS connect initiated to $PERCEPTION_WS_URL (session=$sessionId)")
       }

val Greeting: State = state {

    onEntry {
               restartPerceptionClient(furhat, users.current)

        setAppLanguage(AppLanguage.EN)

        goto(DocumentWaitingToStart)
    }
}