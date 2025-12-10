package furhatos.app.templateadvancedskill.perception

/**
 * Simple in-memory helper for greeting users differently when
 * they return to the robot.
 *
 * This does not replace the backend database; it just keeps track of
 * which user IDs we have already greeted in this skill process.
 */
object UserMemory {

    private val seenUserIds = mutableSetOf<String>()

    /** Returns true if this user ID has been seen before in this process. */
    fun hasSeenBefore(userId: String?): Boolean {
        if (userId.isNullOrBlank()) return false
        return seenUserIds.contains(userId)
    }

    /** Mark this user ID as seen so future greetings can treat them as returning. */
    fun markSeen(userId: String?) {
        if (userId.isNullOrBlank()) return
        seenUserIds.add(userId)
    }
}


