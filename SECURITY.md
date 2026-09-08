# Security Policy

This software is verified for security for official releases and as such does
not make promises about the quality of the product for patches delivered between
releases.

## Security Boundaries

The ML Emulation Layer for Vulkan® is not a sandbox or process-security
boundary. Vulkan® API calls and associated data received by the layers are
expected to come from trusted sources. The layers also rely on the operating
system, Vulkan® Loader, installable client driver (ICD), and caller being
trusted. Weaknesses originating in those components or in a compromised
same-process caller cannot be addressed in the layers.

## Reporting a Vulnerability

Security vulnerabilities may be reported to the Arm® Product Security Incident
Response Team (PSIRT) by sending an email to
[psirt@arm.com](mailto:psirt@arm.com).

For more information visit
https://developer.arm.com/support/arm-security-updates/report-security-vulnerabilities
