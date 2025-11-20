package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.language.AppLanguage
import furhatos.app.templateadvancedskill.language.I18n
import furhatos.app.templateadvancedskill.language.setAppLanguage
import furhatos.app.templateadvancedskill.perception.PerceptionClient
import furhatos.flow.kotlin.*
import furhatos.flow.kotlin.Furhat
import furhatos.flow.kotlin.furhat.audiofeed.AudioFeedListener
import furhatos.flow.kotlin.furhat.camerafeed.CameraFeedListener
import furhatos.flow.kotlin.furhat.camerafeed.FaceData
import furhatos.flow.kotlin.users
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
        client.sendBinary(payload)
    }
}

private object PerceptionAudioListener : AudioFeedListener {
    override fun audioData(data: ByteArray) {
        val client = PerceptionStreaming.activeClient ?: return
        val payload = byteArrayOf(0x02) + data
        client.sendBinary(payload)
    }
}

private const val PERCEPTION_WS_URL = "ws://localhost:8000/ws/perception"
private val perceptionScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

val Greeting: State = state {

    onEntry {
        val sessionId = "furhat-session-" + System.currentTimeMillis()

        val client = PerceptionClient(
            backendWsUrl = PERCEPTION_WS_URL,
            sessionId = sessionId,
            robotId = "furhat-ntnu-01"
        )

        users.current.perceptionClient = client
        client.connect(perceptionScope)
        PerceptionStreaming.start(client)
        PerceptionStreaming.ensureListeners(furhat)

        setAppLanguage(AppLanguage.EN)
        furhat.say(I18n.t("intro_bilingual"))

        val docName = "this topic"
        furhat.say(I18n.t("greet", docName))

        goto(DocumentWaitingToStart)
    }

    onExit {
        users.current.perceptionClient?.close()
        PerceptionStreaming.stop()
    }
}