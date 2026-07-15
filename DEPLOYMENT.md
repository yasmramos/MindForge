# Deployment Guide for Maven Central

This guide walks you through deploying MindForge to Maven Central (OSSRH).

## Prerequisites

1. **Sonatype OSSRH Account**
   - Create account at https://issues.sonatype.org
   - Open a ticket to reserve your group ID: `io.github.yasmramos`

2. **GPG Key Setup**
   ```bash
   # Install GPG
   sudo apt-get install gnupg  # Linux
   brew install gnupg          # macOS
   
   # Generate key pair
   gpg --gen-key
   
   # List keys
   gpg --list-keys
   
   # Export public key
   gpg --armor --export your-email@gmail.com
   ```

3. **Upload Public Key to Keyserver**
   ```bash
   gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID
   ```

4. **Configure Maven Settings**
   
   Edit `~/.m2/settings.xml`:
   ```xml
   <settings>
     <servers>
       <server>
         <id>ossrh</id>
         <username>your-sonatype-username</username>
         <password>your-sonatype-password</password>
       </server>
     </servers>
     <profiles>
       <profile>
         <id>ossrh</id>
         <activation>
           <activeByDefault>true</activeByDefault>
         </activation>
         <properties>
           <gpg.keyname>YOUR_KEY_ID</gpg.keyname>
           <gpg.passphrase>your-gpg-passphrase</gpg.passphrase>
         </properties>
       </profile>
     </profiles>
   </settings>
   ```

## Deployment Steps

### 1. Update Version in pom.xml

Change from SNAPSHOT to release version:
```xml
<version>1.2.2</version>  <!-- Not 1.2.2-SNAPSHOT -->
```

### 2. Build and Deploy to OSSRH Staging

```bash
# Clean and deploy with release profile
mvn clean deploy -P release
```

This will:
- Compile code
- Run tests
- Generate Javadoc
- Create source jars
- Sign artifacts with GPG
- Upload to OSSRH staging repository

### 3. Verify Staging Repository

1. Go to https://oss.sonatype.org
2. Login with your Sonatype credentials
3. Navigate to "Staging Repositories"
4. Find your repository (io.github.yasmramos-xxxx)
5. Click "Close" (this triggers validation)
6. Wait for validation to complete (~2-5 minutes)
7. Click "Release" (this promotes to Maven Central)

### 4. Verify on Maven Central

After ~10-30 minutes, verify your artifact is available:
- https://repo.maven.apache.org/maven2/io/github/yasmramos/mindforge/
- https://search.maven.org/search?q=g:io.github.yasmramos

## Post-Deployment

### Update Version Back to SNAPSHOT

```xml
<version>1.2.3-SNAPSHOT</version>
```

### Add Usage Example to README

```xml
<dependency>
    <groupId>io.github.yasmramos</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.2.2</version>
</dependency>
```

## Troubleshooting

### GPG Signing Errors

```bash
# Ensure gpg-agent is running
gpg-connect-agent reloadagent /bye

# Clear passphrase cache
gpgconf --kill gpg-agent
```

### Permission Denied

Ensure your Sonatype username has permissions for the group ID.

### Validation Failures

Check the staging repository logs for specific errors:
- Missing Javadoc
- Missing sources
- Invalid POM metadata

## GitHub Actions Automation

The project includes `.github/workflows/deploy.yml` for automated deployment on tags:

```bash
# Create and push tag
git tag -a v1.2.2 -m "Release version 1.2.2"
git push origin v1.2.2
```

This triggers automatic deployment if:
- All tests pass
- Running on main branch
- Tag matches version pattern

## Checklist

- [ ] Sonatype account created
- [ ] Group ID reserved
- [ ] GPG key generated and uploaded
- [ ] Maven settings configured
- [ ] Version updated in pom.xml
- [ ] Deployed to staging
- [ ] Staging repository closed and released
- [ ] Verified on Maven Central
- [ ] Version bumped to next SNAPSHOT
- [ ] README updated with usage example

## Resources

- Sonatype OSSRH Guide: https://central.sonatype.org/pages/ossrh-guide.html
- Maven Deploy Plugin: https://maven.apache.org/plugins/maven-deploy-plugin/
- GPG Manual: https://www.gnupg.org/documentation/manuals/gnupg/

---

**Contact**: yasmramos95@gmail.com for support
